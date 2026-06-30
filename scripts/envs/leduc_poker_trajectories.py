"""Random trajectory generator for Leduc Poker SFT training.

Plays games with uniform-random actions against an in-process MCTS opponent
and returns ``(examples, final_reward)`` so that generate_trajectories.py can
apply score-based sampling to bias toward games with positive outcomes.

Produces per-turn tool-calling examples (§3.2/§3.3 of docs/SFT_ALIGNMENT_PLAN.md):
each turn of a game becomes its own standalone
``{"messages": [system, user, assistant(tool_calls=game_action)], "tools": ...}``
example — a stateless [system, user] reconstruction matching
``core/pvp/bot.py::LLMBot._run_turn`` at eval time, with an always-empty memory
block and all 5 tools (4 memory + game_action).

Why random instead of expert: Leduc Poker's tiny state space lets MCTS play
near-optimally, so even a strong heuristic player can rarely win. Generating
random games and filtering/sampling by final reward is more practical than
trying to craft a hand-coded expert that can beat MCTS.

Game mechanics now run on a real ``pyspiel.load_game("leduc_poker", ...)``
state (see docs/PYSPIEL_NATIVE_DATAGEN_PLAN.md) instead of the HTTP env
server; the opponent is an in-process pyspiel MCTSBot instead of the
server's "mcts" option. The policy (uniform-random over legal actions) is
UNCHANGED — only its input (a real ``state.legal_actions()`` list instead of
a regex-parsed action-id list) changed.
"""

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

# rules + empty memory block + tool guidance, mirrors LLMBot._system_prompt
# with a fresh (all-empty) memory state — see §3.2.
_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)


def _build_tool_example(state_desc: str, player_id: int, legal_actions: "list[tuple[int, str]]", action_id: int) -> dict:
    """Build one stateless {system, user, assistant(game_action)} training example."""
    tools = build_pvp_tools([aid for aid, _ in legal_actions])
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state_desc, player_id, legal_actions)},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "game_action", "arguments": json.dumps({"action_id": action_id})}},
            ]},
        ],
        # JSON-encoded: see intercode_dataset.py's _INTERCODE_TOOLS_JSON for why
        # heterogeneous tool `parameters.properties` must not go through
        # Dataset.from_list as a list-of-dicts column.
        "tools": json.dumps(tools_to_openai(tools)),
    }


def generate_random_episode(
    game_id: int,
    max_turn: int = 10,
) -> "tuple[list[dict], float]":
    """
    Run one Leduc Poker game on a real pyspiel state using a random policy
    against an in-process MCTS opponent.

    Returns ``(examples, final_reward)``, where ``examples`` is a list of
    independent per-turn training examples (one per action taken), each shaped
    ``{"messages": [system, user, assistant(tool_calls=[game_action])], "tools": [...]}``.
    ``final_reward`` is the terminal score in [0, 1] (0.0 = max loss, 0.5 =
    draw, 1.0 = max win) from the teacher's seat, used directly by
    generate_trajectories.py as a sampling probability, or 0.0 if ``max_turn``
    teacher decisions elapsed before the game ended.

    ``max_turn`` bounds the number of TEACHER decisions (mirrors the old
    HTTP-server loop, where one /step call = one teacher decision and the
    opponent's reply happened invisibly inside that same call) — not raw
    plies.
    """
    rng = random.Random(game_id)
    teacher_seat = rng.choice((0, 1))
    mcts_simulations = rng.randint(_MCTS_SIMS_MIN, _MCTS_SIMS_MAX)

    game = _AGENT.load_game(_AGENT.generate_params(config_id_for_task_id(game_id)))
    state = game.new_initial_state()
    opponent_bot = make_mcts_bot(game, mcts_simulations, seed=game_id)

    def _resolve_until_teacher_turn() -> bool:
        """Returns False if the game ended (naturally, or because the
        opponent's MCTS search failed -- see mcts_step_or_none; in that case
        we stop the episode here rather than substitute a fabricated
        opponent move and keep going)."""
        while not state.is_terminal() and (state.is_chance_node() or state.current_player() != teacher_seat):
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(outcomes, weights=probs)[0])
            else:
                action = mcts_step_or_none(opponent_bot, state)
                if action is None:
                    _log(f"[leduc_poker_trajectories] Opponent MCTS search failed (game {game_id}), "
                         "truncating episode here")
                    return False
                state.apply_action(action)
        return not state.is_terminal()

    examples: list[dict] = []
    try:
        if not _resolve_until_teacher_turn():
            return [], 0.0

        for _ in range(max_turn):
            legal_actions = [(a, state.action_to_string(teacher_seat, a)) for a in state.legal_actions(teacher_seat)]
            state_desc = state.observation_string(teacher_seat)
            action_id = rng.choice([a for a, _ in legal_actions])
            examples.append(_build_tool_example(state_desc, teacher_seat, legal_actions, action_id))
            state.apply_action(action_id)

            if not _resolve_until_teacher_turn():
                break
        else:
            _log(f"[leduc_poker_trajectories] max_turn={max_turn} reached (game {game_id})")
    except Exception as exc:
        _log(f"[leduc_poker_trajectories] Failed to build episode (game {game_id}): {exc}")
        return examples, 0.0

    final_reward = score_for_player(state, teacher_seat) if state.is_terminal() else 0.0
    return examples, final_reward
