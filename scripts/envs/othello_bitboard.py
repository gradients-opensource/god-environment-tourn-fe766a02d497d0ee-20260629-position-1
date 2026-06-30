"""Bitboard core for Othello: legal moves, flips, and the static evaluation,
all operating on two 64-bit ints (``me``, ``opp``) instead of an 8x8 list of
lists. This exists purely for speed — see othello_minimax.py's module
docstring for why: the list-based implementation only reaches ~3-4 ply of
search in the midgame even with several seconds of budget, because move
generation re-scans the whole board (64 cells x 8 directions) from scratch
on every node. Bitwise ops process a whole direction across the entire board
in one machine instruction-ish step instead of a Python-level nested loop,
which is the standard technique for fast Othello/Reversi engines.

Bit index convention: bit (row*8 + col), row 0 = top, col 0 = 'a'/left —
matches othello_minimax.py's (row, col) convention so the two modules agree
on cell identity without any translation beyond `1 << (row*8+col)`.
"""

import time

_FULL = 0xFFFFFFFFFFFFFFFF
_FILE_A = 0x0101010101010101
_FILE_H = 0x8080808080808080
_NOT_FILE_A = _FULL ^ _FILE_A
_NOT_FILE_H = _FULL ^ _FILE_H

_CORNER_BITS = [1 << 0, 1 << 7, 1 << 56, 1 << 63]  # (0,0) (0,7) (7,0) (7,7)


def _shift_n(b):  return (b >> 8) & _FULL
def _shift_s(b):  return (b << 8) & _FULL
def _shift_e(b):  return ((b & _NOT_FILE_H) << 1) & _FULL
def _shift_w(b):  return ((b & _NOT_FILE_A) >> 1) & _FULL
def _shift_ne(b): return ((b & _NOT_FILE_H) >> 7) & _FULL
def _shift_nw(b): return ((b & _NOT_FILE_A) >> 9) & _FULL
def _shift_se(b): return ((b & _NOT_FILE_H) << 9) & _FULL
def _shift_sw(b): return ((b & _NOT_FILE_A) << 7) & _FULL


_SHIFTS = (_shift_n, _shift_s, _shift_e, _shift_w, _shift_ne, _shift_nw, _shift_se, _shift_sw)


def cell_to_bit(row: int, col: int) -> int:
    return 1 << (row * 8 + col)


def bit_to_cell(bit_index: int) -> "tuple[int, int]":
    return divmod(bit_index, 8)


def to_bitboards(board: "list[list[int]]") -> "tuple[int, int]":
    """``board`` uses othello_minimax.py's ME=1/OPP=-1/EMPTY=0 convention."""
    me = opp = 0
    for r in range(8):
        for c in range(8):
            v = board[r][c]
            if v == 1:
                me |= cell_to_bit(r, c)
            elif v == -1:
                opp |= cell_to_bit(r, c)
    return me, opp


def legal_moves_bb(me: int, opp: int) -> int:
    """Bitmask of empty cells where `me` has a legal move.

    Shifts are inlined (no per-direction function call) and the
    accumulation loop exits as soon as a direction's opponent run ends,
    rather than always unrolling a fixed 5 steps — profiling showed the
    `_shift_*` function-call overhead, not the bitwise math, dominated this
    function's cost (it's by far the hottest function in search).
    """
    occ = me | opp
    empty = _FULL & ~occ
    moves = 0

    c = (me >> 8) & opp
    while c:
        nxt = (c >> 8) & opp
        moves |= (c >> 8) & empty
        c = nxt

    c = (me << 8) & _FULL & opp
    while c:
        nxt = (c << 8) & _FULL & opp
        moves |= (c << 8) & _FULL & empty
        c = nxt

    c = ((me & _NOT_FILE_H) << 1) & _FULL & opp
    while c:
        nxt = ((c & _NOT_FILE_H) << 1) & _FULL & opp
        moves |= ((c & _NOT_FILE_H) << 1) & _FULL & empty
        c = nxt

    c = (me & _NOT_FILE_A) >> 1 & opp
    while c:
        nxt = (c & _NOT_FILE_A) >> 1 & opp
        moves |= (c & _NOT_FILE_A) >> 1 & empty
        c = nxt

    c = ((me & _NOT_FILE_H) >> 7) & opp
    while c:
        nxt = ((c & _NOT_FILE_H) >> 7) & opp
        moves |= ((c & _NOT_FILE_H) >> 7) & empty
        c = nxt

    c = ((me & _NOT_FILE_A) >> 9) & opp
    while c:
        nxt = ((c & _NOT_FILE_A) >> 9) & opp
        moves |= ((c & _NOT_FILE_A) >> 9) & empty
        c = nxt

    c = ((me & _NOT_FILE_H) << 9) & _FULL & opp
    while c:
        nxt = ((c & _NOT_FILE_H) << 9) & _FULL & opp
        moves |= ((c & _NOT_FILE_H) << 9) & _FULL & empty
        c = nxt

    c = ((me & _NOT_FILE_A) << 7) & _FULL & opp
    while c:
        nxt = ((c & _NOT_FILE_A) << 7) & _FULL & opp
        moves |= ((c & _NOT_FILE_A) << 7) & _FULL & empty
        c = nxt

    return moves


def flips_bb(me: int, opp: int, move_bit: int) -> int:
    """Bitmask of opponent discs flipped by placing `me` at `move_bit`.
    Inlined for the same reason as legal_moves_bb."""
    flips = 0

    line = 0
    cand = move_bit >> 8
    while cand & opp:
        line |= cand
        cand = cand >> 8
    if cand & me:
        flips |= line

    line = 0
    cand = (move_bit << 8) & _FULL
    while cand & opp:
        line |= cand
        cand = (cand << 8) & _FULL
    if cand & me:
        flips |= line

    line = 0
    cand = ((move_bit & _NOT_FILE_H) << 1) & _FULL
    while cand & opp:
        line |= cand
        cand = ((cand & _NOT_FILE_H) << 1) & _FULL
    if cand & me:
        flips |= line

    line = 0
    cand = (move_bit & _NOT_FILE_A) >> 1
    while cand & opp:
        line |= cand
        cand = (cand & _NOT_FILE_A) >> 1
    if cand & me:
        flips |= line

    line = 0
    cand = (move_bit & _NOT_FILE_H) >> 7
    while cand & opp:
        line |= cand
        cand = (cand & _NOT_FILE_H) >> 7
    if cand & me:
        flips |= line

    line = 0
    cand = (move_bit & _NOT_FILE_A) >> 9
    while cand & opp:
        line |= cand
        cand = (cand & _NOT_FILE_A) >> 9
    if cand & me:
        flips |= line

    line = 0
    cand = ((move_bit & _NOT_FILE_H) << 9) & _FULL
    while cand & opp:
        line |= cand
        cand = ((cand & _NOT_FILE_H) << 9) & _FULL
    if cand & me:
        flips |= line

    line = 0
    cand = ((move_bit & _NOT_FILE_A) << 7) & _FULL
    while cand & opp:
        line |= cand
        cand = ((cand & _NOT_FILE_A) << 7) & _FULL
    if cand & me:
        flips |= line

    return flips


def apply_move_bb(me: int, opp: int, move_bit: int) -> "tuple[int, int]":
    flips = flips_bb(me, opp, move_bit)
    return me | move_bit | flips, opp & ~flips


def popcount(x: int) -> int:
    return x.bit_count()


# ---------------------------------------------------------------------------
# Evaluation — ports othello_minimax.py's _evaluate exactly, operating on
# bitboards. `_WEIGHTS` is the single source of truth (othello_minimax.py
# imports it from here) so move-ordering and evaluation never drift apart.
# ---------------------------------------------------------------------------

_WEIGHTS = [
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, 5, 1, 1, 5, -2, 10],
    [5, -2, 1, 1, 1, 1, -2, 5],
    [5, -2, 1, 1, 1, 1, -2, 5],
    [10, -2, 5, 1, 1, 5, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100],
]

_WEIGHT_MASKS: "dict[int, int]" = {}
for _r in range(8):
    for _c in range(8):
        _w = _WEIGHTS[_r][_c]
        _WEIGHT_MASKS[_w] = _WEIGHT_MASKS.get(_w, 0) | cell_to_bit(_r, _c)

_DIAG1_MASKS: "dict[int, int]" = {}
for _k in range(-7, 8):
    _m = 0
    for _r in range(8):
        _c = _r - _k
        if 0 <= _c < 8:
            _m |= cell_to_bit(_r, _c)
    _DIAG1_MASKS[_k] = _m

_DIAG2_MASKS: "dict[int, int]" = {}
for _k in range(0, 15):
    _m = 0
    for _r in range(8):
        _c = _k - _r
        if 0 <= _c < 8:
            _m |= cell_to_bit(_r, _c)
    _DIAG2_MASKS[_k] = _m

_ROW_MASKS = tuple(0xFF << (_r * 8) for _r in range(8))
_COL_MASKS = tuple(_FILE_A << _c for _c in range(8))
_DIAG1_MASK_LIST = tuple(_DIAG1_MASKS.values())
_DIAG2_MASK_LIST = tuple(_DIAG2_MASKS.values())
_ALL_CORNERS_MASK = _CORNER_BITS[0] | _CORNER_BITS[1] | _CORNER_BITS[2] | _CORNER_BITS[3]

# Precomputed once: for each of the 8 (corner, edge-direction) pairs, the
# corner's bit and the ordered list of step bits walking outward along that
# edge — avoids recomputing cell_to_bit()/bounds-checking on every
# evaluate_bb call (this runs on every leaf node).
_EDGE_RUNS_BB = tuple(
    (cell_to_bit(cr, cc), tuple(
        cell_to_bit(cr + dr * step, cc + dc * step)
        for step in range(1, 8)
        if 0 <= cr + dr * step < 8 and 0 <= cc + dc * step < 8
    ))
    for (cr, cc), (dr, dc) in [
        ((0, 0), (0, 1)), ((0, 7), (0, -1)),
        ((7, 0), (0, 1)), ((7, 7), (0, -1)),
        ((0, 0), (1, 0)), ((7, 0), (-1, 0)),
        ((0, 7), (1, 0)), ((7, 7), (-1, 0)),
    ]
)


def _edge_run_stable_mask(me: int, opp: int) -> int:
    mask = 0
    for corner_bit, steps in _EDGE_RUNS_BB:
        if me & corner_bit:
            side = me
        elif opp & corner_bit:
            side = opp
        else:
            continue
        for bit in steps:
            if not (side & bit):
                break
            mask |= bit
    return mask


def stable_mask_bb(me: int, opp: int) -> int:
    """Discs that can never be flipped for the rest of the game. Ports
    othello_minimax.py's _stable_mask: fully-enclosed cells (no empty on
    any of their row/col/both diagonals), occupied corners, and same-color
    edge runs anchored to an owned corner."""
    empty = _FULL & ~(me | opp)

    row_full_mask = 0
    for rm in _ROW_MASKS:
        if not (empty & rm):
            row_full_mask |= rm
    col_full_mask = 0
    for cm in _COL_MASKS:
        if not (empty & cm):
            col_full_mask |= cm
    diag1_full_mask = 0
    for dm in _DIAG1_MASK_LIST:
        if not (empty & dm):
            diag1_full_mask |= dm
    diag2_full_mask = 0
    for dm in _DIAG2_MASK_LIST:
        if not (empty & dm):
            diag2_full_mask |= dm
    fully_enclosed = row_full_mask & col_full_mask & diag1_full_mask & diag2_full_mask

    corner_occupied = _ALL_CORNERS_MASK & (me | opp)

    return fully_enclosed | corner_occupied | _edge_run_stable_mask(me, opp)


def evaluate_bb(me: int, opp: int, color: int) -> float:
    """Static evaluation from `color`'s perspective (color is +1 if `me`
    bits represent the side to move, -1 if `opp` bits do — mirrors
    othello_minimax._evaluate's (board, color) convention exactly)."""
    mover, other = (me, opp) if color == 1 else (opp, me)
    occ = me | opp
    empty = _FULL & ~occ

    empties = empty.bit_count()
    phase = (64 - empties) / 64.0

    position_score = 0
    for w, mask in _WEIGHT_MASKS.items():
        position_score += w * ((mask & me).bit_count() - (mask & opp).bit_count())
    position_score *= color

    my_moves = legal_moves_bb(mover, other).bit_count()
    opp_moves = legal_moves_bb(other, mover).bit_count()
    mobility = my_moves - opp_moves

    my_count = mover.bit_count()
    opp_count = other.bit_count()
    total = my_count + opp_count
    parity = 100.0 * (my_count - opp_count) / total if total else 0.0

    occ_with_empty_neighbor = (
        (empty >> 8)
        | ((empty << 8) & _FULL)
        | (((empty & _NOT_FILE_H) << 1) & _FULL)
        | ((empty & _NOT_FILE_A) >> 1)
        | ((empty & _NOT_FILE_H) >> 7)
        | ((empty & _NOT_FILE_A) >> 9)
        | (((empty & _NOT_FILE_H) << 9) & _FULL)
        | (((empty & _NOT_FILE_A) << 7) & _FULL)
    ) & occ
    my_frontier = (occ_with_empty_neighbor & mover).bit_count()
    opp_frontier = (occ_with_empty_neighbor & other).bit_count()
    frontier = opp_frontier - my_frontier

    stable = stable_mask_bb(me, opp)
    my_stable = (stable & mover).bit_count()
    opp_stable = (stable & other).bit_count()
    stability = my_stable - opp_stable

    w_mobility = 15.0 * (1 - phase) + 2.0 * phase
    w_parity = 25.0 * phase
    w_frontier = -2.0 * (1 - phase)
    w_stability = 12.0

    return (
        position_score
        + w_mobility * mobility
        + w_parity * parity
        + w_frontier * frontier
        + w_stability * stability
    )


def eval_components_bb(me: int, opp: int) -> dict:
    """Weighted evaluation components from `me`'s perspective after a move.

    Returns a dict of weighted component scores (same units as evaluate_bb),
    keyed by factor name. Used by reasoning generation to find the dominant
    factor that differentiates the best move from alternatives."""
    occ   = me | opp
    empty = _FULL & ~occ
    phase = (64 - empty.bit_count()) / 64.0

    pos = sum(w * ((mask & me).bit_count() - (mask & opp).bit_count())
              for w, mask in _WEIGHT_MASKS.items())

    my_mob  = legal_moves_bb(me, opp).bit_count()
    opp_mob = legal_moves_bb(opp, me).bit_count()

    total = me.bit_count() + opp.bit_count()
    parity = 100.0 * (me.bit_count() - opp.bit_count()) / total if total else 0.0

    occ_with_empty_neighbor = (
        (empty >> 8) | ((empty << 8) & _FULL)
        | (((empty & _NOT_FILE_H) << 1) & _FULL) | ((empty & _NOT_FILE_A) >> 1)
        | ((empty & _NOT_FILE_H) >> 7) | ((empty & _NOT_FILE_A) >> 9)
        | (((empty & _NOT_FILE_H) << 9) & _FULL) | (((empty & _NOT_FILE_A) << 7) & _FULL)
    ) & occ
    frontier = (occ_with_empty_neighbor & opp).bit_count() - (occ_with_empty_neighbor & me).bit_count()

    stable = stable_mask_bb(me, opp)
    stability = (stable & me).bit_count() - (stable & opp).bit_count()

    return {
        "position":  pos,
        "mobility":  (15.0 * (1 - phase) + 2.0 * phase) * (my_mob - opp_mob),
        "parity":    25.0 * phase * parity,
        "frontier":  -2.0 * (1 - phase) * frontier,
        "stability": 12.0 * stability,
    }


# ---------------------------------------------------------------------------
# Search — same iterative-deepening negamax + alpha-beta + transposition
# table structure as othello_minimax.py, just over bitboards. `color` here
# is the same "absolute ME=1/OPP=-1, which one is to move" convention as
# evaluate_bb's.
# ---------------------------------------------------------------------------

_TIME_CHECK_INTERVAL = 2048
_TT_EXACT, _TT_LOWER, _TT_UPPER = 0, 1, 2


class Timeout(Exception):
    pass


def _move_weight(bit: int) -> int:
    idx = bit.bit_length() - 1
    r, c = divmod(idx, 8)
    return _WEIGHTS[r][c]


def negamax_bb(
    me: int, opp: int, color: int, depth: int, alpha: float, beta: float,
    deadline: float, counter: "list[int]", tt: dict, max_depth: int,
) -> float:
    counter[0] += 1
    if counter[0] % _TIME_CHECK_INTERVAL == 0 and time.monotonic() > deadline:
        raise Timeout()

    orig_alpha, orig_beta = alpha, beta
    mover, other = (me, opp) if color == 1 else (opp, me)
    key = (me, opp, color)
    entry = tt.get(key)
    if entry is not None:
        e_depth, e_score, e_flag = entry
        if e_depth >= depth:
            if e_flag == _TT_EXACT:
                return e_score
            if e_flag == _TT_LOWER:
                alpha = max(alpha, e_score)
            elif e_flag == _TT_UPPER:
                beta = min(beta, e_score)
            if alpha >= beta:
                return e_score

    moves_mask = legal_moves_bb(mover, other)
    if moves_mask == 0:
        if legal_moves_bb(other, mover) == 0:
            score = evaluate_bb(me, opp, color)
            tt[key] = (max_depth, score, _TT_EXACT)
            return score
        if depth == 0:
            score = evaluate_bb(me, opp, color)
            tt[key] = (depth, score, _TT_EXACT)
            return score
        score = -negamax_bb(me, opp, -color, depth - 1, -beta, -alpha, deadline, counter, tt, max_depth)
        tt[key] = (depth, score, _TT_EXACT)
        return score

    if depth == 0:
        score = evaluate_bb(me, opp, color)
        tt[key] = (depth, score, _TT_EXACT)
        return score

    moves = []
    m = moves_mask
    while m:
        lsb = m & (m - 1)
        moves.append(m & ~lsb)
        m = lsb
    moves.sort(key=_move_weight, reverse=True)

    best = -float("inf")
    for move_bit in moves:
        if color == 1:
            new_me, new_opp = apply_move_bb(me, opp, move_bit)
        else:
            new_opp, new_me = apply_move_bb(opp, me, move_bit)
        score = -negamax_bb(new_me, new_opp, -color, depth - 1, -beta, -alpha, deadline, counter, tt, max_depth)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if best <= orig_alpha:
        flag = _TT_UPPER
    elif best >= orig_beta:
        flag = _TT_LOWER
    else:
        flag = _TT_EXACT
    tt[key] = (depth, best, flag)
    return best


def search_bb(
    me: int, opp: int, color: int, moves_mask: int, time_budget: float, max_depth: int,
) -> "tuple[int, int, dict[int, float]]":
    """Iterative-deepening root search.

    Returns (best_move_bit, depth_reached, move_scores) where move_scores maps
    each candidate move_bit to its score from the last completed iteration.
    Callers that only need the move can ignore the extra values.
    """
    moves = []
    m = moves_mask
    while m:
        lsb = m & (m - 1)
        moves.append(m & ~lsb)
        m = lsb
    moves.sort(key=_move_weight, reverse=True)

    best_move = moves[0]
    move_scores: dict[int, float] = {bit: 0.0 for bit in moves}
    deadline = time.monotonic() + time_budget
    counter = [0]
    tt: dict = {}
    depth = 1
    depth_reached = 0
    while depth <= max_depth and time.monotonic() < deadline:
        alpha, beta = -float("inf"), float("inf")
        iter_best_move = moves[0]
        iter_best_score = -float("inf")
        iter_scores: dict[int, float] = {}
        try:
            for move_bit in moves:
                if color == 1:
                    new_me, new_opp = apply_move_bb(me, opp, move_bit)
                else:
                    new_opp, new_me = apply_move_bb(opp, me, move_bit)
                score = -negamax_bb(new_me, new_opp, -color, depth - 1, -beta, -alpha, deadline, counter, tt, max_depth)
                iter_scores[move_bit] = score
                if score > iter_best_score:
                    iter_best_score = score
                    iter_best_move = move_bit
                if iter_best_score > alpha:
                    alpha = iter_best_score
        except Timeout:
            break

        best_move = iter_best_move
        move_scores = iter_scores
        moves.remove(best_move)
        moves.insert(0, best_move)
        depth_reached = depth

        if len(moves) <= 1:
            break
        depth += 1

    return best_move, depth_reached, move_scores
