"""Othello alpha-beta search strategy for SFT trajectory generation.

Parses the ASCII board out of `state_desc`, then runs iterative-deepening
negamax with alpha-beta pruning over a phase-weighted evaluation function
combining (per othello_bot_research.md): a corner/edge/X-square piece-square
table, actual mobility, frontier (potential-mobility) exposure, stable-disc
count, and coin parity (weighted in only as the board fills up).

The actual move generation/eval/search is implemented in othello_bitboard.py
(64-bit-int bitboards instead of an 8x8 list of lists). This module is
responsible for board-text parsing; othello_bitboard.py owns the search.

`time_budget`/`max_depth` are exposed as parameters so the trajectory
generator can vary them per game for diversity.
"""

from envs.othello_bitboard import (
    bit_to_cell,
    cell_to_bit,
    legal_moves_bb,
    search_bb,
    to_bitboards,
)

ME, OPP, EMPTY = 1, -1, 0

# Defaults. Trajectory generator samples per-game values around these.
_TIME_BUDGET = 0.15
_MAX_DEPTH = 14


# --- Board parsing -------------------------------------------------------------------


def _parse_board(state_desc: str) -> list[list[str]]:
    """8x8 grid of '-'/'x'/'o' parsed from the rendered board rows."""
    board = [["-"] * 8 for _ in range(8)]
    for line in state_desc.splitlines():
        tokens = line.split()
        if len(tokens) == 10 and tokens[0] == tokens[-1] and tokens[0].isdigit():
            row = int(tokens[0]) - 1
            if 0 <= row < 8:
                board[row] = tokens[1:9]
    return board


def _my_symbol(state_desc: str) -> str:
    if "Black (x) to play" in state_desc:
        return "x"
    if "White (o) to play" in state_desc:
        return "o"
    return "x"


def _to_internal(raw_board: list[list[str]], my_symbol: str) -> list[list[int]]:
    opp_symbol = "o" if my_symbol == "x" else "x"
    board = [[EMPTY] * 8 for _ in range(8)]
    for r in range(8):
        for c in range(8):
            cell = raw_board[r][c]
            if cell == my_symbol:
                board[r][c] = ME
            elif cell == opp_symbol:
                board[r][c] = OPP
    return board


# --- Entry point -----------------------------------------------------------------------


def choose_action(
    state_desc: str,
    player_id: int,
    legal_actions: list[tuple[int, str]],
    time_budget: float = _TIME_BUDGET,
    max_depth: int = _MAX_DEPTH,
) -> int:
    if len(legal_actions) == 1:
        return legal_actions[0][0]

    raw_board = _parse_board(state_desc)
    board = _to_internal(raw_board, _my_symbol(state_desc))

    label_to_cell: dict[tuple[int, int], int] = {}
    for action_id, label in legal_actions:
        label = label.strip().lower()
        if label == "pass" or len(label) != 2:
            continue
        col = ord(label[0]) - ord("a")
        row = int(label[1]) - 1
        label_to_cell[(row, col)] = action_id

    me, opp = to_bitboards(board)
    moves_mask = legal_moves_bb(me, opp)

    label_cells_mask = 0
    for (r, c) in label_to_cell:
        label_cells_mask |= cell_to_bit(r, c)
    moves_mask &= label_cells_mask
    if moves_mask == 0:
        return legal_actions[0][0]

    best_move_bit, _, _scores = search_bb(me, opp, ME, moves_mask, time_budget, max_depth)
    best_cell = bit_to_cell(best_move_bit.bit_length() - 1)
    return label_to_cell[best_cell]


def choose_action_with_scores(
    state_desc: str,
    player_id: int,
    legal_actions: list[tuple[int, str]],
    time_budget: float = _TIME_BUDGET,
    max_depth: int = _MAX_DEPTH,
) -> "tuple[int, int, int, int, dict[int, float], int]":
    """Like choose_action but also returns search internals for reasoning generation.

    Returns (action_id, me_bb, opp_bb, best_bit, move_scores, depth).
    Falls back to choose_action behaviour (first legal action) when there is only
    one legal move, with empty move_scores and depth=0."""
    if len(legal_actions) == 1:
        return legal_actions[0][0], 0, 0, 0, {}, 0

    raw_board = _parse_board(state_desc)
    board = _to_internal(raw_board, _my_symbol(state_desc))

    label_to_cell: dict[tuple[int, int], int] = {}
    for action_id, label in legal_actions:
        label = label.strip().lower()
        if label == "pass" or len(label) != 2:
            continue
        col = ord(label[0]) - ord("a")
        row = int(label[1]) - 1
        label_to_cell[(row, col)] = action_id

    me, opp = to_bitboards(board)
    moves_mask = legal_moves_bb(me, opp)

    label_cells_mask = 0
    for (r, c) in label_to_cell:
        label_cells_mask |= cell_to_bit(r, c)
    moves_mask &= label_cells_mask
    if moves_mask == 0:
        return legal_actions[0][0], me, opp, 0, {}, 0

    best_bit, depth, move_scores = search_bb(me, opp, ME, moves_mask, time_budget, max_depth)
    best_cell = bit_to_cell(best_bit.bit_length() - 1)
    return label_to_cell[best_cell], me, opp, best_bit, move_scores, depth
