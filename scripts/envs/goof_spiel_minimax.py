"""Lookahead policy for Goofspiel.

Replaces the naive "always bid the card matching the prize value" heuristic
with a per-round zero-sum matrix-game solve. This is possible because
goof_spiel_trajectories.py's local simulator tracks both players' exact
remaining hands as ground truth — only the simultaneous bid itself is
genuinely hidden in real play, not the hand composition. The eval-time LLM
prompt hides the opponent's hand (per
~/sn56-G.O.D-env/docs/GOOFSPIEL_IMPLEMENTATION_NOTES.md), but this *teacher*
policy is free to use it, the same way othello_minimax.py sees the full
board even though the model it's training only sees its own view.

Algorithm per round:
  1. Build the payoff matrix M[i][j] for "I bid card i, opponent bids card j":
       M[i][j] = (prize if i>j else -prize if i<j else 0) + continuation(i, j)
  2. continuation(i, j) is a fast heuristic estimate of the expected margin
     for the *remaining* rounds after this one (see ``_greedy_continuation``)
     — not an exact recursive solve, which is combinatorially infeasible for
     hands up to size 13 (state space blows up: roughly C(N,r)^3 across all
     remaining-round-counts r).
  3. Solve M as a zero-sum game via linear programming (scipy.optimize.linprog)
     to get the Nash equilibrium value and *mixed* strategy over my hand.
  4. Bid the mode of that mixed strategy (deterministic), not a sample from
     it. A live, repeatedly-played agent would want to mix to stay
     unexploitable — but this policy's purpose is to label SFT training
     examples, one discrete (state, action) pair per turn. Sampling would
     mean the *same* state could get a different "correct" action from one
     generated game to the next (most likely to actually collide late-game,
     when the remaining-hand state space is small), which is a contradictory
     and harder-to-learn signal. Mirrors why othello_minimax.choose_action is
     itself a deterministic function of the board — variety across generated
     games comes from varying the game (random task_id/seed), not from
     randomizing the per-move decision.

Exception to the above: against an opponent detected to be playing a fixed,
non-adaptive pattern (``detect_mirror_opponent``), step 4's deterministic
mode is provably exploitable (its minimax guarantee only holds when sampled),
so ``choose_action`` switches to an exact best-response instead of the
generic equilibrium mode — see its docstring.
"""

import numpy as np
from scipy.optimize import linprog


def _solve_zero_sum_game(M: "list[list[float]]") -> "tuple[float, list[float]]":
    """Solve a zero-sum matrix game for the row player.

    Returns (game_value, mixed_strategy) where mixed_strategy[i] is the
    Nash-equilibrium probability of playing row action i.
    """
    Ma = np.asarray(M, dtype=float)
    n, m = Ma.shape
    if n == 1:
        return float(Ma[0].min()), [1.0]

    lo, hi = float(Ma.min()), float(Ma.max())
    # Variables: x_0..x_{n-1} (my mixed strategy), v (game value).
    # maximize v  s.t.  for every opponent action j: sum_i M[i][j]*x_i >= v,
    # sum_i x_i = 1, x_i >= 0.  linprog minimizes, so use c = [0,...,0,-1].
    c = np.zeros(n + 1)
    c[-1] = -1.0

    A_ub = np.zeros((m, n + 1))
    A_ub[:, :n] = -Ma.T
    A_ub[:, n] = 1.0
    b_ub = np.zeros(m)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = [1.0]

    bounds = [(0, 1)] * n + [(lo, hi)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        x = [1.0 / n] * n
        return float(Ma.mean()), x

    x = np.clip(res.x[:n], 0, None)
    x = (x / x.sum()) if x.sum() > 0 else np.ones(n) / n
    return float(res.x[n]), x.tolist()


def _greedy_continuation(my_hand: "list[int]", opp_hand: "list[int]", future: "list[int]") -> float:
    """Cheap O(r log r) proxy for the expected margin of the rounds *after*
    this one: both sides greedily spend their highest remaining card on the
    highest remaining future prize, pairwise, in descending order. Not
    game-theoretically exact (it ignores the hidden-bid dynamics within each
    future round), but a fast, informative signal that rewards conserving
    high-value cards for high-value future prizes rather than burning them
    on the current round's prize."""
    if not future or not my_hand or not opp_hand:
        return 0.0
    me  = sorted(my_hand, reverse=True)
    opp = sorted(opp_hand, reverse=True)
    fut = sorted(future, reverse=True)
    margin = 0.0
    for k, p in enumerate(fut):
        if k >= len(me) or k >= len(opp):
            break
        i, j = me[k], opp[k]
        margin += p if i > j else (-p if i < j else 0.0)
    return margin


def _predict_mirror_bid(hand: "list[int]", prize: int) -> int:
    """What a "bid the prize, nearest-higher/lower fallback" opponent would
    play from `hand` against this prize -- see goof_spiel_trajectories.py's
    ``_bot_mirror``, the exact same rule."""
    if prize in hand:
        return prize
    higher = sorted(c for c in hand if c > prize)
    return higher[0] if higher else max(hand)


def detect_mirror_opponent(
    opp_bid_history: "list[tuple[int, list[int], int]]", min_observations: int = 1,
) -> bool:
    """``opp_bid_history``: (prize, opp_hand_before_round, opp_actual_bid) for
    every round played so far this game. True iff every observed round's
    actual bid exactly matches what a fixed "always bid the prize" opponent
    would have played from its hand at the time -- i.e. nothing seen yet
    rules out that hypothesis.

    This uses the opponent's actual (ground-truth) bid, which the *teacher*
    is free to see for choosing its own move -- same oracle access the
    equilibrium solve always had (see module docstring) -- so detection can
    fire as early as round 1. This is NOT the same thing as deciding when a
    real model would plausibly have noticed and recorded the pattern in its
    own memory notes from its own (much weaker, hidden-bid) observations;
    that's a separate, deliberately later/more conservative judgement made
    by the caller (goof_spiel_trajectories.py) before writing anything to
    memory -- this function only answers "is the hypothesis still true so
    far", not "would the model plausibly know this yet".
    """
    if len(opp_bid_history) < min_observations:
        return False
    return all(
        actual == _predict_mirror_bid(hand_before, prize)
        for prize, hand_before, actual in opp_bid_history
    )


def solve_round(
    my_hand: "list[int]", opp_hand: "list[int]", prize: int, future: "list[int]",
) -> "tuple[list[int], list[float]]":
    """Solve the current round as a zero-sum matrix game.

    Returns ``(my_hand_sorted, mixed_strategy)`` where ``mixed_strategy[k]``
    is the equilibrium probability of bidding ``my_hand_sorted[k]``.
    """
    me  = sorted(my_hand)
    opp = sorted(opp_hand)
    M = [
        [
            (prize if i > j else (-prize if i < j else 0.0))
            + _greedy_continuation(
                [c for c in me if c != i], [c for c in opp if c != j], future,
            )
            for j in opp
        ]
        for i in me
    ]
    _, x = _solve_zero_sum_game(M)
    return me, x


def choose_action(
    my_hand: "list[int]", opp_hand: "list[int]", prize: int, future: "list[int]",
    opp_bid_history: "list[tuple[int, list[int], int]] | None" = None,
) -> int:
    """Deterministically pick the mode of the round's equilibrium mixed
    strategy — see the module docstring for why this isn't a sample.

    A forced move (one card left) returns it directly without solving.

    If ``opp_bid_history`` shows the opponent consistently bidding the prize
    card (``detect_mirror_opponent``), that opponent is exploitable in a way
    the generic equilibrium solve doesn't capture: its equilibrium mixed
    strategy guarantees the game value only when actually SAMPLED, not when
    its mode is played deterministically against a known fixed opponent (see
    goof_spiel_trajectories.py's win-rate breakdown showing this policy
    historically losing to exactly this opponent type). Once detected
    (ground-truth, can fire as early as round 1 — this is the teacher
    choosing its own best move, not something shown to the model), play the
    exact known-optimal counter instead: bid the cheapest card that beats
    the opponent's predicted bid (winning the prize while conserving higher
    cards for future rounds), or sacrifice the cheapest card if no card
    beats it cheaply.
    """
    if len(my_hand) == 1:
        return my_hand[0]
    if opp_bid_history and detect_mirror_opponent(opp_bid_history):
        predicted = _predict_mirror_bid(opp_hand, prize)
        cheap_beat = sorted(c for c in my_hand if c > predicted)
        return cheap_beat[0] if cheap_beat else min(my_hand)
    me, x = solve_round(my_hand, opp_hand, prize, future)
    return me[int(np.argmax(x))]
