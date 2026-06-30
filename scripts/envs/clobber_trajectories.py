"""Trajectory generator for Clobber SFT training.

New PvP env (see ~/sn56-G.O.D-env's docs/clobber_implementation.md) — added
directly against the pyspiel-native generation pattern from
docs/PYSPIEL_NATIVE_DATAGEN_PLAN.md, with no legacy HTTP-server format to
match (unlike the other 5 envs, which were ports of an existing server-backed
generator). See docs/CLOBBER_DATAGEN_PLAN.md for the research behind the
strategy choices here and the v1/v2 split.

Produces per-turn tool-calling examples, one per teacher decision:
``{"messages": [system, user, assistant(tool_calls=[game_action])], "tools": ...}``
— a stateless [system, user] reconstruction matching
``core/pvp/bot.py::LLMBot._run_turn`` at eval time, with an always-empty
memory block (Clobber's full state is always fully visible — no opponent-hand
tracking or memory-worthy hidden information, unlike e.g. liar_dice/gin_rummy).

Policy: iterative-deepening alpha-beta search over real pyspiel state
(clobber_minimax.choose_action), with a mobility-difference heuristic — NOT
material count, which doesn't work for Clobber's "last player able to move
wins" win condition. Opponent: in-process MCTS (make_mcts_bot), matching the
real eval-time baseline opponent (eval_payload_extra: {"opponent": "mcts"}).

v1 deliberately skips the "double view" (teacher + MCTS-opponent both
recorded as training examples) that othello_trajectories.py does — single
view only, to keep this first version simple to verify. See
docs/CLOBBER_DATAGEN_PLAN.md's "Deferred to v2" list.
"""

import json
import random

from envs.clobber_minimax import choose_action
from envs.pvp_format import build_full_system_prompt
from envs.pvp_format import build_pvp_tools
from envs.pvp_format import build_user_prompt
from envs.pvp_format import tools_to_openai
from envs.pvp_game_engine import ClobberAgent
from envs.pvp_game_engine import config_id_for_task_id
from envs.pvp_game_engine import make_mcts_bot
from envs.pvp_game_engine import mcts_step_or_none
from envs.pvp_game_engine import score_for_player
from envs.shared_env import _log

_GAME_NAME = "clobber"
_AGENT = ClobberAgent()

# Per-game opponent strength: random MCTS simulation budget (mirrors
# othello_trajectories.py's range).
_MCTS_SIMS_MIN = 50
_MCTS_SIMS_MAX = 100

# Per-game search budget for choose_action. Clobber's boards (<=30 cells) are
# much smaller than Othello's 8x8 -- measured throughput puts depth 8 from
# the OPENING position (the worst case) at well under the high end of this
# range; see docs/CLOBBER_DATAGEN_PLAN.md for the benchmark.
_TIME_BUDGET_RANGE = (0.3, 1.0)
_MAX_DEPTH_RANGE = (14, 16)

# Game length is hard-capped at cells-1 (<=29 for the largest board, since
# every move captures exactly one piece) -- 40 is generous enough that this
# essentially never truncates, so final_reward is almost always a real
# terminal outcome rather than the max_turn-cutoff fallback.
_DEFAULT_MAX_TURN = 40

# rules + empty memory block + tool guidance, mirrors LLMBot._system_prompt
# with a fresh (all-empty) memory state.
_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)


def _build_tool_example(
    state_desc: str, player_id: int, legal_actions: "list[tuple[int, str]]", action_id: int,
) -> dict:
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


def generate_heuristic_episode(
    game_id: int,
    max_turn: int = _DEFAULT_MAX_TURN,
) -> "tuple[list[dict], float]":
    """
    Run one Clobber game on a real pyspiel state using alpha-beta search
    (clobber_minimax) against an in-process MCTS opponent.

    Returns ``(examples, final_reward)``, where ``examples`` is a list of
    independent per-turn training examples (one per teacher decision), and
    ``final_reward`` is the terminal score in [0, 1] (0.0 = loss, 1.0 = win
    -- Clobber has no draws, win/loss is determined purely by who runs out
    of legal moves first) from the teacher's seat, or 0.0 if ``max_turn``
    was hit before the game ended (practically never, see _DEFAULT_MAX_TURN).
    Returns ``([], 0.0)`` on a mid-game error.
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

    # max_turn bounds the number of TEACHER decisions, not raw plies (same
    # convention as the other 4 strictly-alternating/no-chance-node envs).
    # Clobber strictly alternates seats every ply (no passes -- a player
    # with no legal move has lost, the game is terminal), so each loop
    # iteration below is exactly one teacher turn plus, if the game isn't
    # over, the one opponent reply that follows it.
    try:
        for _ in range(max_turn):
            if state.is_terminal():
                break
            cur = state.current_player()
            legal_actions = [(a, state.action_to_string(cur, a)) for a in state.legal_actions(cur)]
            state_desc = _AGENT.format_state(state, cur)

            if cur == teacher_seat:
                action_id = choose_action(state, cur, time_budget, max_depth)
                examples.append(_build_tool_example(state_desc, cur, legal_actions, action_id))
                state.apply_action(action_id)
            else:
                # Should not happen on a fresh iteration (seats strictly
                # alternate so the opponent's reply is consumed below in the
                # same iteration as the teacher's move that preceded it) --
                # guard anyway rather than silently mis-attributing a turn.
                action_id = mcts_step_or_none(opponent_bot, state)
                if action_id is None:
                    _log(f"[clobber_trajectories] Opponent MCTS search failed (game {game_id}), "
                         "truncating episode here")
                    break
                state.apply_action(action_id)
                continue

            if state.is_terminal():
                break
            opp_action_id = mcts_step_or_none(opponent_bot, state)
            if opp_action_id is None:
                _log(f"[clobber_trajectories] Opponent MCTS search failed (game {game_id}), "
                     "truncating episode here")
                break
            state.apply_action(opp_action_id)
        else:
            _log(f"[clobber_trajectories] max_turn={max_turn} reached (game {game_id})")
    except Exception as exc:
        _log(f"[clobber_trajectories] Failed to build episode (game {game_id}): {exc}")
        return [], 0.0

    final_reward = score_for_player(state, teacher_seat) if state.is_terminal() else 0.0
    return examples, final_reward
