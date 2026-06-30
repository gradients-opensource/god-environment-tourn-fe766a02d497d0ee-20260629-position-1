"""Liar's Dice trajectory-generation policy.

Scores each legal action against a posterior over the opponent's hand,
inferred from their observed bid history, rather than a flat uniform prior
over their dice.

Action-id encoding: for total dice N, bid (quantity Q, face F) ->
id = (Q-1)*6 + (F-1), for Q in 1..N, F in 1..6; "Liar" -> id = N*6.
Legal actions given a current bid id `c` are every bid id > c, plus Liar
(if c is not None).
"""

import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache

_SAMPLING_TEMPERATURE = 0.01

# Candidate opponent softmax temperatures, marginalized over with a uniform
# prior when scoring how likely an observed opponent bid is under each
# candidate hand.
_OPPONENT_TEMPERATURE_GRID = (0.01, 0.05, 0.1, 0.3)

_RE_DICE = re.compile(r"Your dice:\s*\[([^\]]+)\]")
_RE_TOTAL = re.compile(r"Total dice in game:\s*(\d+)")
_RE_BID = re.compile(r'Current bid:\s*"(\d+)-(\d+)"')


@dataclass
class Bid:
    quantity: int
    face: int


def _decode(action_id: int) -> Bid:
    q, f = divmod(action_id, 6)
    return Bid(q + 1, f + 1)


def _bid_id(bid: Bid) -> int:
    return (bid.quantity - 1) * 6 + (bid.face - 1)


@lru_cache(maxsize=None)
def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _counts(dice: list[int]) -> tuple[int, ...]:
    c = [0] * 6
    for d in dice:
        c[d - 1] += 1
    return tuple(c)


@lru_cache(maxsize=None)
def _enumerate_hands(n: int) -> tuple[tuple[tuple[int, ...], float], ...]:
    """All face-count vectors (c1..c6) summing to n, with multinomial prior weights."""
    def gen(remaining: int, k: int):
        if k == 1:
            yield (remaining,)
            return
        for i in range(remaining + 1):
            for rest in gen(remaining - i, k - 1):
                yield (i,) + rest

    out = []
    for c in gen(n, 6):
        coeff = math.factorial(n)
        for ci in c:
            coeff //= math.factorial(ci)
        out.append((c, coeff * (1 / 6) ** n))
    return tuple(out)


def _prob_bid_true(bid: Bid, counts: tuple[int, ...], n_hidden: int) -> float:
    """P(bid is true) given a known face-count vector and n_hidden other iid-uniform dice."""
    if bid.face == 6:
        known = counts[5]
        p_hit = 1 / 6
    else:
        known = counts[bid.face - 1] + counts[5]
        p_hit = 2 / 6
    still_needed = bid.quantity - known
    if still_needed <= 0:
        return 1.0
    if n_hidden <= 0:
        return 0.0
    return _binom_sf(still_needed, n_hidden, p_hit)


def _softmax_weights(scores: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        best = max(range(len(scores)), key=lambda i: scores[i])
        return [1.0 if i == best else 0.0 for i in range(len(scores))]
    scaled = [s / temperature for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _legal_ids(context_id: "int | None", total_dice: int) -> list[int]:
    liar_id = total_dice * 6
    if context_id is None:
        return list(range(liar_id))
    return list(range(context_id + 1, liar_id)) + [liar_id]


def _opponent_action_score(
    action_id: int, hand: tuple[int, ...], n_us: int, context_bid: "Bid | None", liar_id: int
) -> float:
    if action_id == liar_id:
        return 1.0 - _prob_bid_true(context_bid, hand, n_us)
    return _prob_bid_true(_decode(action_id), hand, n_us)


def _posterior_hands(
    observations: list[tuple["int | None", int]], n_opp: int, n_us: int, total_dice: int
) -> list[tuple[tuple[int, ...], float]]:
    """Posterior over the opponent's face-count vector given their bid history."""
    liar_id = total_dice * 6
    hands = _enumerate_hands(n_opp)
    posterior = []
    for hand, prior in hands:
        if not observations:
            posterior.append((hand, prior))
            continue

        obs_scores = []
        obs_idx = []
        for context_id, chosen_id in observations:
            context_bid = _decode(context_id) if context_id is not None else None
            legal = _legal_ids(context_id, total_dice)
            scores = [_opponent_action_score(a, hand, n_us, context_bid, liar_id) for a in legal]
            obs_scores.append(scores)
            obs_idx.append(legal.index(chosen_id))

        # Marginalize over candidate opponent temperatures (shared across this
        # episode's observations) rather than assuming a single fixed value.
        mix_likelihood = 0.0
        for temp in _OPPONENT_TEMPERATURE_GRID:
            likelihood = 1.0
            for scores, idx in zip(obs_scores, obs_idx):
                likelihood *= _softmax_weights(scores, temp)[idx]
            mix_likelihood += likelihood
        mix_likelihood /= len(_OPPONENT_TEMPERATURE_GRID)

        posterior.append((hand, prior * mix_likelihood))
    total = sum(w for _, w in posterior)
    if total <= 0:
        return [(hand, prior) for hand, prior in hands]
    return [(hand, w / total) for hand, w in posterior]


def _prob_bid_true_posterior(
    bid: Bid, our_counts: tuple[int, ...], posterior: list[tuple[tuple[int, ...], float]]
) -> float:
    if bid.face == 6:
        our_known = our_counts[5]
    else:
        our_known = our_counts[bid.face - 1] + our_counts[5]
    still_needed = bid.quantity - our_known
    if still_needed <= 0:
        return 1.0
    total = 0.0
    for opp_counts, p in posterior:
        opp_known = opp_counts[5] if bid.face == 6 else opp_counts[bid.face - 1] + opp_counts[5]
        if opp_known >= still_needed:
            total += p
    return total


# Per-episode state (module-level; reset via reset_episode() at the start of
# each hand — see that function for why the old dice-equality heuristic alone
# wasn't a safe reset trigger).
_state: "dict | None" = None


def reset_episode() -> None:
    """Clear the accumulated bid history. Call once at the start of each hand.

    The previous reset trigger was "_state is None or observed dice changed",
    which silently carried a finished hand's bid history into a new hand that
    happens to deal identical dice to the same worker process (likely over a
    long generation run) — corrupting that new hand's posterior with an
    unrelated opponent's bids. An explicit per-episode reset removes that
    failure mode entirely; the dice-equality check below is kept only as a
    defensive fallback.
    """
    global _state
    _state = None


def _phrase(action_id: int, total_dice: int) -> str:
    """Human-readable description of a bid/Liar-call action, e.g. "bid 3-4"."""
    if action_id == total_dice * 6:
        return "called Liar"
    bid = _decode(action_id)
    return f"bid {bid.quantity}-{bid.face}"


def choose_action(state_desc: str, player_id: int, legal_actions: list[tuple[int, str]]) -> tuple[int, str]:
    """Pick a legal action; also return a one-line note for working memory.

    The returned ``memory_note`` records what the opponent just bid and what
    we bid in response. The caller (liar_dice_trajectories.py) is expected to
    append it to the model's own working-memory slot every turn so it
    accumulates into a visible bid-history transcript across the hand.

    This matters because the scoring below conditions on `_state["observations"]`
    — the full bid history of this hand, tracked across calls — which is NOT
    present in `state_desc` (only the single most recent bid is ever shown,
    matching the real eval-time state text — see
    core/pvp/agents.py::LiarsDiceAgent.format_state, `bid_parts[-1]`). Without
    surfacing that history back into the model's own input via memory, the
    label this function produces would depend on information invisible to the
    training row it labels (docs/SFT_ALIGNMENT_PLAN.md §5.1).
    """
    global _state

    dice_match = _RE_DICE.search(state_desc)
    our_dice = [int(x.strip()) for x in dice_match.group(1).split(",")] if dice_match else []

    total_match = _RE_TOTAL.search(state_desc)
    total_dice = int(total_match.group(1)) if total_match else len(our_dice) * 2

    bid_match = _RE_BID.search(state_desc)
    current_bid_id = _bid_id(Bid(int(bid_match.group(1)), int(bid_match.group(2)))) if bid_match else None

    if _state is None or _state["our_dice"] != our_dice:
        _state = {"our_dice": our_dice, "last_bid_id": None, "observations": []}

    if current_bid_id is not None and current_bid_id != _state["last_bid_id"]:
        _state["observations"].append((_state["last_bid_id"], current_bid_id))

    if len(legal_actions) == 1:
        chosen_id = legal_actions[0][0]
    else:
        n_opp = total_dice - len(our_dice)
        posterior = _posterior_hands(_state["observations"], n_opp, len(our_dice), total_dice)
        our_counts = _counts(our_dice)

        current_bid = _decode(current_bid_id) if current_bid_id is not None else None
        scores = []
        for action_id, label in legal_actions:
            if label.strip().lower() == "liar":
                scores.append(1.0 - _prob_bid_true_posterior(current_bid, our_counts, posterior))
            else:
                scores.append(_prob_bid_true_posterior(_decode(action_id), our_counts, posterior))

        weights = _softmax_weights(scores, _SAMPLING_TEMPERATURE)
        chosen_id = random.choices(legal_actions, weights=weights, k=1)[0][0]

    opp_part = f"opponent {_phrase(current_bid_id, total_dice)}" if current_bid_id is not None else "no prior bid"
    memory_note = f"{opp_part}; I {_phrase(chosen_id, total_dice)}."

    _state["last_bid_id"] = chosen_id
    return chosen_id, memory_note
