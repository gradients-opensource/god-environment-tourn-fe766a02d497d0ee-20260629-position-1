"""Simple random trajectory generator for Leduc Poker SFT training."""

import json
import random

from envs.pvp_format import build_full_system_prompt
from envs.pvp_format import build_pvp_tools
from envs.pvp_format import build_user_prompt
from envs.pvp_format import tools_to_openai
from envs.pvp_game_engine import config_id_for_task_id
from envs.pvp_game_engine import LeducPokerAgent
from envs.pvp_game_engine import make_mcts_bot
from envs.pvp_game_engine import mcts_step_or_none
from envs.pvp_game_engine import score_for_player
from envs.shared_env import _log

_GAME_NAME = "leduc_poker"
_AGENT = LeducPokerAgent()

_MCTS_SIMS_MIN = 50
_MCTS_SIMS_MAX = 50

_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)


def _build_tool_example(state_desc: str, player_id: int, legal_actions: "list[tuple[int, str]]", action_id: int) -> dict:
    tools = build_pvp_tools([aid for aid, _ in legal_actions])
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state_desc, player_id, legal_actions)},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "game_action", "arguments": json.dumps({"action_id": action_id})}},
            ]},
        ],
        "tools": json.dumps(tools_to_openai(tools)),
    }


def generate_simple_episode(
    game_id: int,
    max_turn: int = 10,
) -> "tuple[list[dict], float]":
    rng = random.Random(game_id)
    teacher_seat = rng.choice((0, 1))
    mcts_simulations = rng.randint(_MCTS_SIMS_MIN, _MCTS_SIMS_MAX)

    game = _AGENT.load_game(_AGENT.generate_params(config_id_for_task_id(game_id)))
    state = game.new_initial_state()
    opponent_bot = make_mcts_bot(game, mcts_simulations, seed=game_id)

    def _resolve_until_teacher_turn() -> bool:
        while not state.is_terminal() and (state.is_chance_node() or state.current_player() != teacher_seat):
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(outcomes, weights=probs)[0])
            else:
                action = mcts_step_or_none(opponent_bot, state)
                if action is None:
                    _log(f"[leduc_poker_simple] Opponent MCTS search failed (game {game_id}), "
                         "truncating episode here")
                    return False
                state.apply_action(action)
        return not state.is_terminal()

    examples: list[dict] = []
    try:
        if not _resolve_until_teacher_turn():
            return [], 0.0

        for _ in range(max_turn):
            all_actions = [(a, state.action_to_string(teacher_seat, a)) for a in state.legal_actions(teacher_seat)]
            candidates = [(a, label) for a, label in all_actions if label.lower() != "fold"]
            if not candidates:
                candidates = all_actions
            state_desc = _AGENT.format_state(state, teacher_seat)
            action_id = rng.choice(candidates)[0]
            examples.append(_build_tool_example(state_desc, teacher_seat, all_actions, action_id))
            state.apply_action(action_id)

            if not _resolve_until_teacher_turn():
                break
        else:
            _log(f"[leduc_poker_simple] max_turn={max_turn} reached (game {game_id})")
    except Exception as exc:
        _log(f"[leduc_poker_simple] Failed to build episode (game {game_id}): {exc}")
        return examples, 0.0

    final_reward = score_for_player(state, teacher_seat) if state.is_terminal() else 0.0
    return examples, final_reward
