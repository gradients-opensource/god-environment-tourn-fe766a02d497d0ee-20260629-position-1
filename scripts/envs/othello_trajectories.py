"""Trajectory generator for Othello SFT training.

Produces per-turn tool-calling examples (§3.4 of docs/SFT_ALIGNMENT_PLAN.md):
each turn of a game becomes its own standalone
``{"messages": [system, user, assistant(tool_calls=game_action)], "tools": ...}``
example — a stateless [system, user] reconstruction matching
``core/pvp/bot.py::LLMBot._run_turn`` at eval time, with an always-empty memory
block and all 5 tools (4 memory + game_action).

Policy: iterative-deepening alpha-beta search (othello_minimax.choose_action).
UNCHANGED from before — it parses ``state_desc`` via plain-text regex/grid
parsing (``_parse_board``/``_my_symbol`` in othello_minimax.py), and
pyspiel's real othello ``observation_string`` renders the exact same board
format ("Black (x) to play:\\n  a b c d e f g h  \\n1 - - - ...", confirmed by
direct testing), so the search needed zero changes — only how that text and
the legal-action list reach it changed, from HTTP+regex to direct pyspiel
state (see docs/PYSPIEL_NATIVE_DATAGEN_PLAN.md).

This also drops the "double view" reconstruction tricks (othello_board.py's
``apply_action_symbols``/``legal_actions_symbols``/``render_board``) that
existed only to rebuild the MCTS opponent's (state, legal_actions) from
"our" state_desc + ``info.last_opponent_action`` because the HTTP server
never showed it to us directly. With a real pyspiel state, the opponent's
turn is just the *next* turn in the same loop — its (state, legal_actions)
is read directly off live state, no reconstruction needed. othello_board.py
itself is left in place (unused by this file now, but still depended on by
othello_minimax_selfcheck.py's regression test).

Othello-specific quirk (see G.O.D's core/pvp/agents.py OthelloAgent.format_state):
the eval-time state description is prefixed with "You play x (Black)." or
"You play o (White)." based on player_id (0=Black/x, 1=White/o) — small models
otherwise misjudge their own colour. _build_tool_example_from_parts reproduces
this prefix (pvp_game_engine.OthelloAgent.format_state does the same thing,
but this file builds the prefix itself to match the search's prefix-less
input exactly, same as before).
"""

import json
import random

from envs.othello_bitboard import apply_move_bb, bit_to_cell, cell_to_bit, eval_components_bb, legal_moves_bb
from envs.othello_minimax import choose_action_with_scores
from envs.pvp_format import build_full_system_prompt
from envs.pvp_format import build_pvp_tools
from envs.pvp_format import build_user_prompt
from envs.pvp_format import tools_to_openai
from envs.pvp_game_engine import config_id_for_task_id
from envs.pvp_game_engine import make_mcts_bot
from envs.pvp_game_engine import mcts_step_or_none
from envs.pvp_game_engine import OthelloAgent
from envs.pvp_game_engine import score_for_player
from envs.shared_env import _log

_GAME_NAME = "othello"
_AGENT = OthelloAgent()

_CORNERS: frozenset[tuple[int, int]] = frozenset({(0, 0), (0, 7), (7, 0), (7, 7)})
_X_SQUARE_CORNER: dict[tuple[int, int], tuple[int, int]] = {
    (1, 1): (0, 0), (1, 6): (0, 7), (6, 1): (7, 0), (6, 6): (7, 7),
}
_FACTOR_DESC: dict[str, str] = {
    "position":  "better square placement than",
    "mobility":  "leaves opponent fewer replies than",
    "parity":    "more discs on the board than",
    "frontier":  "fewer exposed discs than",
    "stability": "more stable discs than",
}


def _cell_label(r: int, c: int) -> str:
    return f"{chr(ord('a') + c)}{r + 1}"


def _generate_reasoning(
    me_bb: int,
    opp_bb: int,
    legal_actions: list[tuple[int, str]],
    move_scores: dict[int, float],
    best_bit: int,
) -> str:
    """One-sentence reasoning grounded in which evaluation factor drove best move."""
    best_rc = bit_to_cell(best_bit.bit_length() - 1)
    best_label = _cell_label(*best_rc)

    candidates = []
    for action_id, label in legal_actions:
        lbl = label.strip().lower()
        if lbl == "pass" or len(lbl) != 2:
            continue
        col = ord(lbl[0]) - ord("a")
        row = int(lbl[1]) - 1
        move_bit = cell_to_bit(row, col)
        if move_bit not in move_scores:
            continue
        new_me, new_opp = apply_move_bb(me_bb, opp_bb, move_bit)
        # evaluate from opp's perspective after our move (lower = better for us)
        comps = eval_components_bb(new_opp, new_me)
        is_corner = (row, col) in _CORNERS
        is_x = (row, col) in _X_SQUARE_CORNER
        corner_open = is_x and not ((me_bb | opp_bb) & cell_to_bit(*_X_SQUARE_CORNER[(row, col)]))
        candidates.append({
            "label": lbl,
            "move_bit": move_bit,
            "score": move_scores[move_bit],
            "comps": comps,
            "is_corner": is_corner,
            "x_risk": corner_open,
        })

    corners  = [c for c in candidates if c["is_corner"]]
    risky    = [c for c in candidates if c["x_risk"]]
    playable = [c for c in candidates if not c["x_risk"] and not c["is_corner"]]
    best_cand = next((c for c in candidates if c["label"] == best_label), None)

    parts = []

    if corners:
        names = "/".join(c["label"] for c in corners)
        parts.append(f"{names} is a corner — take it.")

    for r in risky:
        cr = _X_SQUARE_CORNER[(ord(r["label"][1]) - ord("1"), ord(r["label"][0]) - ord("a"))]
        parts.append(f"{r['label']} diagonal to open corner {_cell_label(*cr)} — skip.")

    pool = corners if corners else playable
    if len(pool) >= 2 and best_cand and not corners:
        others = [c for c in pool if c["label"] != best_label]
        if others:
            runner = max(others, key=lambda c: c["score"])
            deltas = {f: best_cand["comps"][f] - runner["comps"][f] for f in _FACTOR_DESC}
            dominant = min(deltas, key=lambda f: deltas[f])
            parts.append(f"{best_label} {_FACTOR_DESC[dominant]} {runner['label']}.")
    elif len(pool) == 1 and not corners:
        parts.append("Only safe option.")

    parts.append(f"→ {best_label}.")
    return " ".join(parts)

# Per-game opponent strength: random MCTS simulation budget (mirrors
# gin_rummy_trajectories.py's range, widened to add more variety).
_MCTS_SIMS_MIN = 50
_MCTS_SIMS_MAX = 100

# Per-game search budget for choose_action: small jitter around a target
# average. othello_minimax searches via othello_bitboard's bitboard core
# (~7-10x more nodes/sec than the old list-based search). (0.20, 0.35)
# reaches ~5 ply in the midgame instead of ~3, at ~1.5-1.7x today's
# generation wall-clock time — see prior session notes / git history for the
# benchmark this range is based on.
_TIME_BUDGET_RANGE = (0.20, 0.35)
_MAX_DEPTH_RANGE = (12, 16)

# Toggle: whether to include the opponent (MCTS) view in generated data at
# all. Sampled by score (generate_trajectories.py's score**power filter)
# rather than a hard win-only gate, same as the "ours" view.
_INCLUDE_OPPONENT_VIEW = True

# rules + empty memory block + tool guidance, mirrors LLMBot._system_prompt
# with a fresh (all-empty) memory state — see §3.2.
_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)


def _build_tool_example_from_parts(
    state_desc: str,
    player_id: int,
    legal_actions: list[tuple[int, str]],
    action_id: int,
    reasoning: "str | None" = None,
) -> dict:
    """Build one stateless {system, user, assistant(game_action)} training example."""
    colour = "x (Black)" if player_id == 0 else "o (White)"
    state_desc = f"You play {colour}.\n{state_desc}"
    tools = build_pvp_tools([aid for aid, _ in legal_actions])
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state_desc, player_id, legal_actions)},
            {"role": "assistant", "content": reasoning, "tool_calls": [
                {"type": "function", "function": {"name": "game_action", "arguments": json.dumps({"action_id": action_id})}},
            ]},
        ],
        # JSON-encoded: see intercode_dataset.py's _INTERCODE_TOOLS_JSON for why
        # heterogeneous tool `parameters.properties` must not go through
        # Dataset.from_list as a list-of-dicts column.
        "tools": json.dumps(tools_to_openai(tools)),
    }


def generate_heuristic_episode(
    game_id: int,
    max_turn: int = 70,
) -> "list[tuple[list[dict], float]]":
    """
    Run one Othello game on a real pyspiel state using alpha-beta search
    (othello_minimax) against an in-process MCTS opponent.

    Produces two "views" of the game, both sampled by generate_trajectories.py's
    score**power filter:
      - ours: one example per turn the teacher played, ``(examples_ours, final_reward)``.
      - opponent: one example per MCTS response move, read directly off live
        state, ``(examples_opp, 1.0 - final_reward)``.
    ``_INCLUDE_OPPONENT_VIEW`` is a hard on/off switch on top of that
    sampling — when False, ``examples_opp`` is always emptied regardless of
    outcome.

    ``final_reward`` is the terminal score in [0, 1] (0.0 = loss, 0.5 = draw,
    1.0 = win) from the teacher's seat, or 0.0 if ``max_turn`` was hit before
    the game ended.

    Returns ``[(examples_ours, final_reward), (examples_opp_or_empty, 1.0 - final_reward)]``
    on success, or ``[([], 0.0), ([], 1.0)]`` on a mid-game error (any
    examples collected so far are discarded along with their unknown outcome).
    """
    rng = random.Random(game_id)
    teacher_seat = rng.choice((0, 1))
    mcts_simulations = rng.randint(_MCTS_SIMS_MIN, _MCTS_SIMS_MAX)
    time_budget = rng.uniform(*_TIME_BUDGET_RANGE)
    max_depth = rng.randint(*_MAX_DEPTH_RANGE)

    game = _AGENT.load_game(_AGENT.generate_params(config_id_for_task_id(game_id)))
    state = game.new_initial_state()
    _AGENT.setup_initial_state(state, seed=game_id)
    opponent_bot = make_mcts_bot(game, mcts_simulations, seed=game_id)

    examples: list[dict] = []
    opp_examples: list[dict] = []

    # max_turn bounds the number of TEACHER decisions (mirrors the old
    # HTTP-server loop, where one /step call = one teacher decision and the
    # opponent's reply happened invisibly inside that same call) — not raw
    # plies. Othello strictly alternates seats every ply (a forced "pass" is
    # itself a legal move), so each loop iteration below is exactly one
    # teacher turn plus, if the game isn't over, the one opponent reply that
    # follows it.
    try:
        for _ in range(max_turn):
            if state.is_terminal():
                break
            cur = state.current_player()
            legal_actions = [(a, state.action_to_string(cur, a)) for a in state.legal_actions(cur)]
            state_desc = state.observation_string(cur)

            if cur == teacher_seat:
                action_id, me_bb, opp_bb, best_bit, move_scores, _ = choose_action_with_scores(
                    state_desc, cur, legal_actions, time_budget, max_depth
                )
                reasoning = (
                    _generate_reasoning(me_bb, opp_bb, legal_actions, move_scores, best_bit)
                    if move_scores else None
                )
                examples.append(_build_tool_example_from_parts(state_desc, cur, legal_actions, action_id, reasoning))
                state.apply_action(action_id)
            else:
                # Should not happen on a fresh iteration (seats strictly
                # alternate so the opponent's reply is consumed below in the
                # same iteration as the teacher's move that preceded it) —
                # guard anyway rather than silently mis-attributing a turn.
                action_id = mcts_step_or_none(opponent_bot, state)
                if action_id is None:
                    _log(f"[othello_trajectories] Opponent MCTS search failed (game {game_id}), "
                         "truncating episode here")
                    break
                opp_examples.append(_build_tool_example_from_parts(state_desc, cur, legal_actions, action_id))
                state.apply_action(action_id)
                continue

            if state.is_terminal():
                break
            opp_cur = state.current_player()
            opp_legal = [(a, state.action_to_string(opp_cur, a)) for a in state.legal_actions(opp_cur)]
            opp_state_desc = state.observation_string(opp_cur)
            opp_action_id = mcts_step_or_none(opponent_bot, state)
            if opp_action_id is None:
                _log(f"[othello_trajectories] Opponent MCTS search failed (game {game_id}), "
                     "truncating episode here")
                break
            opp_examples.append(_build_tool_example_from_parts(opp_state_desc, opp_cur, opp_legal, opp_action_id))
            state.apply_action(opp_action_id)
        else:
            _log(f"[othello_trajectories] max_turn={max_turn} reached (game {game_id})")
    except Exception as exc:
        _log(f"[othello_trajectories] Failed to build episode (game {game_id}): {exc}")
        return [([], 0.0), ([], 1.0)]

    final_reward = score_for_player(state, teacher_seat) if state.is_terminal() else 0.0
    opp_view = (opp_examples, 1.0 - final_reward) if _INCLUDE_OPPONENT_VIEW else ([], 1.0 - final_reward)
    return [(examples, final_reward), opp_view]
