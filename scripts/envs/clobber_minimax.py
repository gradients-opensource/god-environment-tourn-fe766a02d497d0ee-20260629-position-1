"""Clobber alpha-beta search strategy for SFT trajectory generation.

Runs iterative-deepening negamax with alpha-beta pruning directly over real
pyspiel.State (clone-based) -- unlike othello_minimax.py, there's no legacy
HTTP-server text format to match here, since this is a brand-new env built
straight against pyspiel: no board-text parsing layer is needed.

Leaf evaluation is mobility difference: len(legal_actions(mover)) -
len(legal_actions(opponent)). Clobber's win condition is "last player able
to move wins", not material count (a pile of pieces with no adjacent enemy
is dead weight -- it can't capture or be captured), so mobility, not piece
count, is what the heuristic tracks. See docs/CLOBBER_DATAGEN_PLAN.md for
the research behind this choice and measured depth/throughput numbers.

`mover` is threaded explicitly through the search rather than read off
state.current_player(), because that's undefined at a terminal state and
Clobber strictly alternates players every ply (no passes, no multi-phase
turns, no chance nodes once the game starts) -- so the mover for any node
is just 1 - parent's mover, regardless of what state.current_player() would
report.

`time_budget`/`max_depth` are exposed as parameters (rather than fixed
module constants) so the trajectory generator can vary them per game, same
as othello_minimax.
"""

import time

import pyspiel

_TIME_BUDGET = 0.5
_MAX_DEPTH = 16

# Check the deadline every this-many recursive calls rather than every call --
# time.monotonic() is cheap but not free, and at ~300k nodes/sec (measured,
# see docs/CLOBBER_DATAGEN_PLAN.md) a per-node syscall would be a real tax.
_DEADLINE_CHECK_INTERVAL = 256


class _TimeUp(Exception):
    pass


def _mobility_diff(state: pyspiel.State, mover: int) -> float:
    opp = 1 - mover
    return float(len(state.legal_actions(mover)) - len(state.legal_actions(opp)))


def _ordered_actions(state: pyspiel.State, mover: int, actions: "list[int]") -> "list[int]":
    """Sort candidate moves by a cheap 1-ply lookahead so alpha-beta prunes
    harder. Winning moves first; otherwise prefer moves that leave the
    opponent with the worst mobility relative to the mover's own."""
    opp = 1 - mover

    def key(a: int) -> float:
        child = state.clone()
        child.apply_action(a)
        if child.is_terminal():
            r = child.returns()[mover]
            return -1e9 if r > 0 else (1e9 if r < 0 else 0.0)
        return float(len(child.legal_actions(opp)) - len(child.legal_actions(mover)))

    return sorted(actions, key=key)


def _negamax(
    state: pyspiel.State, mover: int, depth: int, alpha: float, beta: float,
    deadline: float, counter: "list[int]",
) -> float:
    counter[0] += 1
    if counter[0] % _DEADLINE_CHECK_INTERVAL == 0 and time.monotonic() > deadline:
        raise _TimeUp()

    if state.is_terminal():
        return state.returns()[mover]
    if depth == 0:
        return _mobility_diff(state, mover)

    best = -1e18
    for a in _ordered_actions(state, mover, state.legal_actions(mover)):
        child = state.clone()
        child.apply_action(a)
        val = -_negamax(child, 1 - mover, depth - 1, -beta, -alpha, deadline, counter)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def _search_root(
    state: pyspiel.State, mover: int, depth: int, deadline: float, counter: "list[int]",
) -> "tuple[int, float]":
    actions = state.legal_actions(mover)
    ordered = _ordered_actions(state, mover, actions)
    best_action = ordered[0]
    best_val = -1e18
    alpha, beta = -1e18, 1e18
    for a in ordered:
        child = state.clone()
        child.apply_action(a)
        val = -_negamax(child, 1 - mover, depth - 1, -beta, -alpha, deadline, counter)
        if val > best_val:
            best_val = val
            best_action = a
        if best_val > alpha:
            alpha = best_val
    return best_action, best_val


def choose_action(
    state: pyspiel.State,
    player_id: int,
    time_budget: float = _TIME_BUDGET,
    max_depth: int = _MAX_DEPTH,
) -> int:
    """Iterative-deepening alpha-beta over real pyspiel state.

    Searches depth 1, 2, 3, ... up to max_depth, stopping as soon as
    time_budget is exhausted -- keeps the best move found at the last fully
    completed depth (standard iterative deepening: a depth that times out
    partway through is discarded, not trusted).
    """
    legal = state.legal_actions(player_id)
    if len(legal) == 1:
        return legal[0]

    deadline = time.monotonic() + time_budget
    best_action = legal[0]
    depth = 1
    while depth <= max_depth:
        counter = [0]
        try:
            action, _ = _search_root(state, player_id, depth, deadline, counter)
        except _TimeUp:
            break
        best_action = action
        if time.monotonic() > deadline:
            break
        depth += 1
    return best_action
