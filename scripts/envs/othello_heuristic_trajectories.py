"""Trajectory generator for Othello SFT training — heuristic teacher variant.

Uses a simple 3-rule positional heuristic as the teacher policy instead of
the alpha-beta minimax search in othello_trajectories.py. The motivation:
the minimax-trained model was found to lose to this very heuristic in eval,
meaning the model fails to learn deep-search moves but *can* learn the simple
rules the heuristic applies. Teaching the model to mimic the heuristic is a
strictly easier imitation target that directly plugs the observed gap.

Heuristic rules (ported from examples/mock_strategies/othello_heuristic_fix.py):
  Rule 1 — Corner priority: take any available corner immediately.
  Rule 2 — Dangerous-square filter: exclude X-squares then C-squares while
            at least one alternative remains.
  Rule 3 — Positional weight table: among remaining candidates, pick the
            highest-weight cell (corners=100, edges=10, interior=1-5,
            C-squares=-20, X-squares=-50).

Reasoning: each teacher example carries a short chain-of-thought that
directly names the rule that fired, so the model can learn the policy
reasoning and not just the action. This is more learnable than the minimax
factor-explanations (position/mobility/frontier/stability) which require
global search the model cannot perform at inference time.

Opponent: in-process MCTS (make_mcts_bot) with 50-100 sims — same as
othello_trajectories.py. Score filter: sample_by_score=True (same default
as clobber) keeps more winning-game examples.

Interface: generate_heuristic_episode(game_id, max_turn) returns
  [(examples_ours, final_reward), (examples_opp, 1 - final_reward)]
identical to othello_trajectories.py so generate_trajectories.py needs no
changes.

The minimax generator (othello_trajectories.py) is preserved unchanged;
this module replaces it as the default in sft_env_configs.py.
"""

import json
import random

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

# ---------------------------------------------------------------------------
# Heuristic — ported from examples/mock_strategies/othello_heuristic_fix.py
# ---------------------------------------------------------------------------

# Action IDs: action = row * 8 + col (row 0 = rank 1, col 0 = file a).
_CORNERS   = frozenset({0, 7, 56, 63})          # a1, h1, a8, h8
_X_SQUARES = frozenset({9, 14, 49, 54})          # b2, g2, b7, g7
_C_SQUARES = frozenset({1, 8, 6, 15, 57, 48, 62, 55})  # b1/a2, g1/h2, b8/a7, g8/h7
_PASS_ACTION = 64                                 # OpenSpiel forced-pass token

_WEIGHTS = [
    [100, -20,  10,   5,   5,  10, -20, 100],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [ 10,  -2,   5,   1,   1,   5,  -2,  10],
    [  5,  -2,   1,   1,   1,   1,  -2,   5],
    [  5,  -2,   1,   1,   1,   1,  -2,   5],
    [ 10,  -2,   5,   1,   1,   5,  -2,  10],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [100, -20,  10,   5,   5,  10, -20, 100],
]


def _weight(action_id: int) -> int:
    return _WEIGHTS[action_id // 8][action_id % 8]


def _heuristic_choose(legal_action_ids: "list[int]") -> "tuple[int, str]":
    """Apply the 3-rule heuristic, returning (action_id, reasoning).

    reasoning is a short human-readable sentence explaining which rule fired.
    """
    if legal_action_ids == [_PASS_ACTION]:
        return _PASS_ACTION, "No board moves available — forced pass."

    board_actions = [a for a in legal_action_ids if a != _PASS_ACTION]

    # Rule 1: take a corner if available.
    for a in board_actions:
        if a in _CORNERS:
            lbl = chr(ord("a") + a % 8) + str(a // 8 + 1)
            return a, f"Corner {lbl} is available — taking it."

    # Rule 2: filter dangerous squares.
    safe = [a for a in board_actions if a not in _X_SQUARES and a not in _C_SQUARES]
    candidates = safe or [a for a in board_actions if a not in _X_SQUARES] or list(board_actions)
    skipped_x = [a for a in board_actions if a in _X_SQUARES]
    skipped_c = [a for a in board_actions if a in _C_SQUARES and a not in _X_SQUARES]

    # Rule 3: highest positional weight among candidates.
    chosen = max(candidates, key=_weight)
    chosen_lbl = chr(ord("a") + chosen % 8) + str(chosen // 8 + 1)

    # Forced fallback: had to play a bad square.
    if chosen in _X_SQUARES:
        return chosen, f"No safe moves — forced to play X-square {chosen_lbl}."
    if chosen in _C_SQUARES:
        return chosen, f"No X-free safe moves — forced to play C-square {chosen_lbl}."

    # Normal path: name what was skipped and what was chosen.
    parts = []
    if skipped_x:
        xlbls = ", ".join(chr(ord("a") + a % 8) + str(a // 8 + 1) for a in skipped_x)
        parts.append(f"Avoid X-squares ({xlbls}).")
    if skipped_c:
        clbls = ", ".join(chr(ord("a") + a % 8) + str(a // 8 + 1) for a in skipped_c)
        parts.append(f"Avoid C-squares ({clbls}).")
    parts.append(f"Play {chosen_lbl}.")

    return chosen, " ".join(parts)


# ---------------------------------------------------------------------------
# Opponent MCTS config — same range as othello_trajectories.py
# ---------------------------------------------------------------------------

_MCTS_SIMS_MIN = 25
_MCTS_SIMS_MAX = 75

# Toggle: include opponent (lower-sim MCTS) view as additional training examples.
_INCLUDE_OPPONENT_VIEW = False

_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)


def _build_tool_example(
    state_desc: str,
    player_id: int,
    legal_actions: "list[tuple[int, str]]",
    action_id: int,
    reasoning: "str | None" = None,
) -> dict:
    """Build one stateless {system, user, assistant(game_action)} training example."""
    tools = build_pvp_tools([aid for aid, _ in legal_actions])
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state_desc, player_id, legal_actions)},
            {"role": "assistant", "content": reasoning, "tool_calls": [
                {"type": "function", "function": {"name": "game_action", "arguments": json.dumps({"action_id": action_id})}},
            ]},
        ],
        "tools": json.dumps(tools_to_openai(tools)),
    }


def generate_heuristic_episode(
    game_id: int,
    max_turn: int = 70,
) -> "list[tuple[list[dict], float]]":
    """
    Run one Othello game: heuristic teacher vs in-process MCTS opponent.

    Returns [(examples_ours, final_reward), (examples_opp, 1 - final_reward)].
    examples_ours: one per teacher (heuristic) decision, with short reasoning.
    examples_opp: one per MCTS opponent decision (no reasoning, content=None).
    Returns [([], 0.0), ([], 1.0)] on a mid-game error.
    """
    rng = random.Random(game_id)
    teacher_seat = rng.choice((0, 1))
    mcts_simulations = rng.randint(_MCTS_SIMS_MIN, _MCTS_SIMS_MAX)

    game = _AGENT.load_game(_AGENT.generate_params(config_id_for_task_id(game_id)))
    state = game.new_initial_state()
    _AGENT.setup_initial_state(state, seed=game_id)
    opponent_bot = make_mcts_bot(game, mcts_simulations, seed=game_id)

    examples: list[dict] = []
    opp_examples: list[dict] = []

    try:
        for _ in range(max_turn):
            if state.is_terminal():
                break
            cur = state.current_player()
            legal_actions = [(a, state.action_to_string(cur, a)) for a in state.legal_actions(cur)]
            state_desc = _AGENT.format_state(state, cur)

            if cur == teacher_seat:
                action_id, reasoning = _heuristic_choose([aid for aid, _ in legal_actions])
                examples.append(_build_tool_example(state_desc, cur, legal_actions, action_id, reasoning))
                state.apply_action(action_id)
            else:
                # Guard: opponent's turn appears here only if it starts first.
                action_id = mcts_step_or_none(opponent_bot, state)
                if action_id is None:
                    _log(f"[othello_heuristic_trajectories] Opponent MCTS failed (game {game_id}), truncating")
                    break
                opp_examples.append(_build_tool_example(state_desc, cur, legal_actions, action_id))
                state.apply_action(action_id)
                continue

            if state.is_terminal():
                break
            opp_cur = state.current_player()
            opp_legal = [(a, state.action_to_string(opp_cur, a)) for a in state.legal_actions(opp_cur)]
            opp_state_desc = _AGENT.format_state(state, opp_cur)
            opp_action_id = mcts_step_or_none(opponent_bot, state)
            if opp_action_id is None:
                _log(f"[othello_heuristic_trajectories] Opponent MCTS failed (game {game_id}), truncating")
                break
            opp_examples.append(_build_tool_example(opp_state_desc, opp_cur, opp_legal, opp_action_id))
            state.apply_action(opp_action_id)
        else:
            _log(f"[othello_heuristic_trajectories] max_turn={max_turn} reached (game {game_id})")
    except Exception as exc:
        _log(f"[othello_heuristic_trajectories] Failed to build episode (game {game_id}): {exc}")
        return [([], 0.0), ([], 1.0)]

    final_reward = score_for_player(state, teacher_seat) if state.is_terminal() else 0.0
    opp_view = (opp_examples, 1.0 - final_reward) if _INCLUDE_OPPONENT_VIEW else ([], 1.0 - final_reward)
    return [(examples, final_reward), opp_view]
