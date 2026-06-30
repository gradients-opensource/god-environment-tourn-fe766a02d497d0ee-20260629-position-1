"""Board-text <-> internal-grid helpers for Othello's "double view" SFT data.

Used to reconstruct the MCTS opponent's per-turn (state, legal_actions) from
"our" turn's state_desc + chosen action_id + ``info.last_opponent_action``,
so the opponent's move can also become a training example (see
othello_trajectories.py).

Action-id encoding (matches the mcts-api env server / OpenSpiel Othello):
``action_id = row*8 + col`` (0-indexed row/col, 0-63), e.g. 44 -> "e6"
(row=5, col=4). ``action_id == 64`` is "pass" (no legal board moves).
"""

from envs.othello_bitboard import apply_move_bb, bit_to_cell, cell_to_bit, legal_moves_bb, to_bitboards
from envs.othello_minimax import _to_internal

PASS_ACTION_ID = 64


def _action_to_cell(action_id: int) -> tuple[int, int]:
    return action_id // 8, action_id % 8


def _cell_to_label(row: int, col: int) -> str:
    return f"{chr(ord('a') + col)}{row + 1}"


def apply_action_symbols(raw_board: list[list[str]], action_id: int, symbol: str) -> list[list[str]]:
    """Apply `symbol`'s move (action_id, possibly PASS) to a symbol board ('x'/'o'/'-')."""
    if action_id == PASS_ACTION_ID:
        return [row[:] for row in raw_board]

    opp_symbol = "o" if symbol == "x" else "x"
    board = _to_internal(raw_board, symbol)
    me, opp = to_bitboards(board)
    row, col = _action_to_cell(action_id)
    new_me, new_opp = apply_move_bb(me, opp, cell_to_bit(row, col))

    new_raw = [["-"] * 8 for _ in range(8)]
    for idx in range(64):
        r, c = divmod(idx, 8)
        bit = 1 << idx
        if new_me & bit:
            new_raw[r][c] = symbol
        elif new_opp & bit:
            new_raw[r][c] = opp_symbol
    return new_raw


def legal_actions_symbols(raw_board: list[list[str]], symbol: str) -> list[tuple[int, str]]:
    """Legal actions for `symbol` on a symbol board, as [(action_id, label), ...]."""
    board = _to_internal(raw_board, symbol)
    me, opp = to_bitboards(board)
    moves_mask = legal_moves_bb(me, opp)
    if not moves_mask:
        return [(PASS_ACTION_ID, "pass")]
    out = []
    m = moves_mask
    while m:
        lsb = m & (m - 1)
        bit = m & ~lsb
        m = lsb
        r, c = bit_to_cell(bit.bit_length() - 1)
        out.append((r * 8 + c, _cell_to_label(r, c)))
    return out


def render_board(raw_board: list[list[str]], to_play_symbol: str) -> str:
    """Render a symbol board back into the env's "Current State" text block."""
    colour = "Black" if to_play_symbol == "x" else "White"
    header = "  a b c d e f g h  "
    lines = [f"{colour} ({to_play_symbol}) to play:", header]
    for r in range(8):
        row_cells = " ".join(raw_board[r])
        lines.append(f"{r + 1} {row_cells} {r + 1}")
    lines.append(header)
    return "\n".join(lines)
