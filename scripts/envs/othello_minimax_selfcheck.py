"""Standalone regression check for othello_minimax's X-square guard.

Runs full self-play games through the exact same code path
othello_trajectories.py drives in production (render_board ->
legal_actions_symbols -> choose_action -> apply_action_symbols, all via
rendered state_desc strings, no env server required) and asserts the teacher
never plays a genuinely unsafe X-square (b2/g2/b7/g7) — i.e. one where the
opponent could actually take the adjacent corner on their very next move —
when a non-trapping legal move was also available that turn. Forced X-square
plays (every legal move that turn is unsafe) are expected and not flagged.
_is_unsafe_x_square below mirrors othello_minimax._filter_x_squares's precise
per-move check exactly (simulate the move, see if the corner is then in the
opponent's legal-move set) rather than the older, blunter "corner is merely
empty" rule — that blunter rule blocked plenty of perfectly safe X-squares
too (see git history), so this check would otherwise flag intentional,
correct behavior as a false "violation".

This exists because the underlying failure mode reached a production PvP
tournament once already (see G.O.D's
examples/othello_strategy_analysis_5EEaxgnm.md): a miner checkpoint trained
on this teacher's SFT output played an X-square in every one of 10 games and
held 0-1 corners. _filter_x_squares (othello_minimax.py) makes the avoidance
a hard rule instead of a soft eval term; this script is the cheap way to
catch a future regression of that guard before burning a 200-game
tournament run on it again.

Usage: python othello_minimax_selfcheck.py [--games N] [--time-budget S]
"""

import argparse
import random
import sys

from envs.othello_bitboard import apply_move_bb, cell_to_bit, legal_moves_bb, to_bitboards
from envs.othello_board import apply_action_symbols, legal_actions_symbols, render_board
from envs.othello_minimax import _X_SQUARE_CORNER, _to_internal, choose_action

def _initial_board() -> list[list[str]]:
    board = [["-"] * 8 for _ in range(8)]
    board[3][3], board[3][4] = "o", "x"
    board[4][3], board[4][4] = "x", "o"
    return board


def _label_to_cell(label: str) -> tuple[int, int]:
    return int(label[1]) - 1, ord(label[0]) - ord("a")


def _is_unsafe_x_square(board: list[list[int]], row: int, col: int) -> bool:
    """Mirrors othello_minimax._filter_x_squares's precise check: unsafe iff
    the corner is still empty AND the opponent's legal-move set after this
    move actually includes that corner (not merely "corner is empty")."""
    corner = _X_SQUARE_CORNER.get((row, col))
    if corner is None:
        return False
    if board[corner[0]][corner[1]] != 0:  # EMPTY
        return False
    me, opp = to_bitboards(board)
    move_bit = cell_to_bit(row, col)
    new_me, new_opp = apply_move_bb(me, opp, move_bit)
    opp_replies = legal_moves_bb(new_opp, new_me)
    return bool(opp_replies & cell_to_bit(*corner))


def _play_game(time_budget: float, max_depth: int) -> dict:
    raw_board = _initial_board()
    symbol = "x"  # Black moves first.
    consecutive_passes = 0
    violations: list[tuple[str, str]] = []  # (symbol, label) of any avoidable unsafe X-square play.
    x_square_plays = 0

    for _ in range(120):
        legal = legal_actions_symbols(raw_board, symbol)
        if legal[0][1] != "pass":
            board = _to_internal(raw_board, symbol)
            state_desc = render_board(raw_board, symbol)
            action_id = choose_action(state_desc, 0, legal, time_budget, max_depth)
            label = next(lbl for aid, lbl in legal if aid == action_id)

            row, col = _label_to_cell(label)
            if _is_unsafe_x_square(board, row, col):
                x_square_plays += 1
                had_safe_alternative = any(
                    not _is_unsafe_x_square(board, *_label_to_cell(alt_label))
                    for _, alt_label in legal
                    if alt_label != label
                )
                if had_safe_alternative:
                    violations.append((symbol, label))

            raw_board = apply_action_symbols(raw_board, action_id, symbol)
            consecutive_passes = 0
        else:
            consecutive_passes += 1

        symbol = "o" if symbol == "x" else "x"
        if consecutive_passes >= 2:
            break

    corners = {(0, 0): "a1", (0, 7): "h1", (7, 0): "a8", (7, 7): "h8"}
    held = {sym: [label for (r, c), label in corners.items() if raw_board[r][c] == sym] for sym in "xo"}

    return {"violations": violations, "x_square_plays": x_square_plays, "corners": held}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--time-budget", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    all_violations: list[tuple[int, str, str]] = []
    total_x_square_plays = 0

    for game_id in range(args.games):
        result = _play_game(args.time_budget, args.max_depth)
        total_x_square_plays += result["x_square_plays"]
        for symbol, label in result["violations"]:
            all_violations.append((game_id, symbol, label))
        print(
            f"game {game_id}: x-square plays={result['x_square_plays']} "
            f"x corners={result['corners']['x']} o corners={result['corners']['o']} "
            f"violations={len(result['violations'])}"
        )

    print(f"\n{args.games} games, {total_x_square_plays} total X-square plays, "
          f"{len(all_violations)} avoidable unsafe X-square plays")
    if all_violations:
        for game_id, symbol, label in all_violations:
            print(f"  VIOLATION game={game_id} symbol={symbol} move={label}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
