"""Expert trajectory generator for Liar's Dice SFT training.

Produces per-turn tool-calling examples (§3.2/§5.1 of docs/SFT_ALIGNMENT_PLAN.md):
each turn of a game becomes its own standalone
``{"messages": [system, user, assistant(tool_calls=[working_memory_append,
game_action])], "tools": ...}`` example, matching
``core/pvp/bot.py::LLMBot._run_turn`` at eval time.

Game mechanics now run on a real ``pyspiel.load_game("liars_dice", ...)``
state (see docs/PYSPIEL_NATIVE_DATAGEN_PLAN.md) instead of the hand-rolled
local dice/bid simulator this file used previously -- that simulator was a
reasonable substitute when the choice was "hand-rolled" vs. "slow HTTP env
server", but pyspiel removes that tradeoff entirely (same correctness
guarantee as the real engine, no network, and faster: a real game step is a
C++ call). The teacher policy (``liar_dice_policy.choose_action``) and the
local opponent policy (``_bot_choose_action``, below) are UNCHANGED -- only
how game state reaches them changed, from hand-rolled dice arrays + manually
tracked bid context to direct pyspiel state. ``liar_dice_policy.choose_action``
parses its ``state_desc`` input via regex (``Your dice: [...]`` /
``Total dice in game: N`` / ``Current bid: "Q-F"``), and
``pvp_game_engine.LiarsDiceAgent.format_state`` (a verbatim port of
``core/pvp/agents.py::LiarsDiceAgent.format_state``) renders exactly that
format from real pyspiel state -- confirmed by direct testing, so the policy
needed zero changes.

Opponent: ``_bot_choose_action`` scores each legal action with the exact
same probability-of-bid-being-true formula the teacher's own posterior
inference assumes opponents use (``liar_dice_policy._opponent_action_score``
-- P(bid true) given the bot's own dice and the other player's hidden dice),
then samples via softmax at a temperature randomized per game from
``_OPPONENT_TEMPERATURE_CHOICES``. This keeps bids "reasonable" (grounded in
the bot's actual dice, never nonsense) while being deliberately looser than
the teacher's own near-deterministic mode (temperature=0.01) -- matching
"not too strong" -- and varying game to game. Kept exactly as before
(verified empirically that it's not too easy for the teacher to beat -- see
prior session notes); this plan only swaps the mechanics it reads from, not
its decision logic.

Turn order: which seat (0 or 1) the teacher plays is randomized per game_id,
matching the real env server's ``llm_player_id`` semantics -- replicated here
exactly as before: if assigned seat 1, the local opponent bids once before
the main loop (mirrors real pyspiel's liars_dice always handing the first
decision to player 0).

Unlike the other envs' SFT generators, the system prompt's memory block here
is NOT always-empty: liar_dice_policy.choose_action's posterior over the
opponent's hand depends on this hand's full bid history, but the rendered
state text only ever shows the single most recent bid (matching eval time --
core/pvp/agents.py::LiarsDiceAgent.format_state). To keep the label a
function of something actually visible in the row, each turn's assistant
message records a one-line bid-history note via working_memory_append
*before* calling game_action, and the next turn's system prompt is rebuilt
from the accumulated notes -- exactly the "edit memory, then commit a move"
contract LLMBot offers at eval time (§5.1).
"""

import json
import random

import pyspiel

from envs.liar_dice_policy import _counts
from envs.liar_dice_policy import _decode
from envs.liar_dice_policy import _legal_ids
from envs.liar_dice_policy import _opponent_action_score
from envs.liar_dice_policy import _softmax_weights
from envs.liar_dice_policy import choose_action
from envs.liar_dice_policy import reset_episode
from envs.pvp_format import build_pvp_tools
from envs.pvp_format import build_tool_system_prompt
from envs.pvp_format import build_user_prompt
from envs.pvp_format import default_memories
from envs.pvp_format import memory_block
from envs.pvp_format import TOOL_GUIDANCE
from envs.pvp_format import tools_to_openai
from envs.pvp_game_engine import LiarsDiceAgent
from envs.pvp_models import MemoryArea
from envs.pvp_models import MemoryOp
from envs.pvp_tools import memory_tool_name
from envs.shared_env import _log

_GAME_NAME = "liars_dice"
_AGENT = LiarsDiceAgent()

# Fixed by LiarsDiceAgent.generate_params (players=2, numdice=5) regardless of
# config_id -- same fixed setup the previous hand-rolled simulator used.
_DICE_PER_PLAYER = 5
_NUM_PLAYERS = 2
_TOTAL_DICE = _DICE_PER_PLAYER * _NUM_PLAYERS

# Local opponent's softmax temperature over its probability-based action
# scores, randomized per game for variety (and to keep it "not too strong" --
# the teacher's own near-deterministic mode uses 0.01; these are deliberately
# looser while still weighted toward higher-probability, reasonable bids).
_OPPONENT_TEMPERATURE_CHOICES = (0.1, 0.2, 0.3, 0.5)

_WORKING_LOG_SLOT = 1
_WORKING_APPEND_TOOL = memory_tool_name(MemoryArea.WORKING, MemoryOp.APPEND)

# Rules text only -- cached so the per-turn system prompt (which varies with
# accumulated memory, see generate_expert_episode) doesn't re-parse
# pvp_game_prompts.yml from disk every turn of every game.
_RULES_PROMPT = build_tool_system_prompt(_GAME_NAME)


def _dice_for(state: pyspiel.State, player_id: int) -> "list[int]":
    """Read a player's own dice back off live pyspiel state.

    Mirrors the digit-extraction LiarsDiceAgent.format_state does internally
    (pyspiel's information_state_string for this game is "{dice_digits}
    [bid]", e.g. "23455" or "23455 6-1") -- duplicated here (rather than
    imported) because it's the one piece of LiarsDiceAgent's internals this
    file needs raw, not pre-rendered to text.
    """
    info = state.information_state_string(player_id)
    dice_part = info.split()[0] if info else ""
    return [int(d) for d in dice_part if d.isdigit()]


def _bot_choose_action(
    opp_dice: "list[int]", n_us: int, total_dice: int, context_id: "int | None",
    rng: random.Random, temperature: float,
) -> int:
    """Local opponent: score each legal action by P(bid true) given its own
    dice (the same formula liar_dice_policy's posterior inference assumes
    opponents use), then sample via softmax -- see module docstring."""
    liar_id = total_dice * 6
    legal = _legal_ids(context_id, total_dice)
    opp_counts = _counts(opp_dice)
    context_bid = _decode(context_id) if context_id is not None else None
    scores = [_opponent_action_score(a, opp_counts, n_us, context_bid, liar_id) for a in legal]
    weights = _softmax_weights(scores, temperature)
    return rng.choices(legal, weights=weights, k=1)[0]


def _build_tool_example_from_parts(
    system_prompt: str,
    state_desc: str,
    player_id: int,
    legal_actions: "list[tuple[int, str]]",
    action_id: int,
    memory_note: str,
) -> dict:
    """Build one {system, user, assistant(memory + game_action)} training example.

    ``system_prompt`` carries this hand's accumulated bid-history note (built
    by the caller from earlier turns), not an always-empty memory block --
    see the module docstring. The assistant message edits memory *and*
    commits a move in the same response, exactly as LLMBot._run_turn allows
    at eval time ("any number of memory tool calls plus exactly one
    game_action call").
    """
    tools = build_pvp_tools([aid for aid, _ in legal_actions])
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(state_desc, player_id, legal_actions)},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {
                    "name": _WORKING_APPEND_TOOL,
                    "arguments": json.dumps({"slot": _WORKING_LOG_SLOT, "content": memory_note}),
                }},
                {"type": "function", "function": {
                    "name": "game_action",
                    "arguments": json.dumps({"action_id": action_id}),
                }},
            ]},
        ],
        # JSON-encoded: see intercode_dataset.py's _INTERCODE_TOOLS_JSON for why
        # heterogeneous tool `parameters.properties` must not go through
        # Dataset.from_list as a list-of-dicts column.
        "tools": json.dumps(tools_to_openai(tools)),
    }


def generate_expert_episode(
    game_id: int,
    max_turn: int = 30,
) -> list[dict]:
    """
    Run one Liar's Dice game on a real pyspiel state using
    liar_dice_policy's opponent-modeling expert policy against a
    randomly-chosen (per game_id) local opponent.

    Returns a list of independent per-turn training examples (one per action
    taken by the expert), each shaped ``{"messages": [system, user,
    assistant(tool_calls=[working_memory_append, game_action])], "tools": [...]}``.
    The system prompt's memory block carries this hand's bid-history note
    accumulated from earlier turns (see module docstring / §5.1) instead of
    the always-empty block other envs use. Returns ``[]`` on failure.
    """
    rng = random.Random(game_id)
    player_id = rng.choice((0, 1))
    opp_id = 1 - player_id
    temperature = rng.choice(_OPPONENT_TEMPERATURE_CHOICES)

    game = _AGENT.load_game(_AGENT.generate_params(config_id=0))
    state = game.new_initial_state()
    while state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        state.apply_action(rng.choices(outcomes, weights=probs)[0])
    opp_dice = _dice_for(state, opp_id)

    reset_episode()
    memories = default_memories()
    examples: list[dict] = []
    context_id: "int | None" = None

    try:
        # If the teacher is assigned the second-moving seat, the opponent
        # bids first -- pyspiel's liars_dice always hands the first decision
        # to player 0 after dealing, matching the real env server's
        # llm_player_id=1 behavior this previously replicated. Never a Liar
        # call here since context_id is None.
        if player_id == 1:
            context_id = _bot_choose_action(opp_dice, _DICE_PER_PLAYER, _TOTAL_DICE, None, rng, temperature)
            state.apply_action(context_id)

        for _ in range(max_turn):
            state_desc = _AGENT.format_state(state, player_id)
            legal_actions = [
                (a, state.action_to_string(player_id, a)) for a in state.legal_actions(player_id)
            ]
            system_prompt = "\n\n".join([_RULES_PROMPT, memory_block(memories), TOOL_GUIDANCE])
            action_id, memory_note = choose_action(state_desc, player_id, legal_actions)
            examples.append(_build_tool_example_from_parts(
                system_prompt, state_desc, player_id, legal_actions, action_id, memory_note,
            ))
            memories[MemoryArea.WORKING].append(_WORKING_LOG_SLOT, memory_note)
            state.apply_action(action_id)

            if action_id == _TOTAL_DICE * 6:
                break  # teacher called Liar; hand over
            context_id = action_id

            opp_action = _bot_choose_action(opp_dice, _DICE_PER_PLAYER, _TOTAL_DICE, context_id, rng, temperature)
            state.apply_action(opp_action)
            if opp_action == _TOTAL_DICE * 6:
                break  # opponent called Liar; hand over
            context_id = opp_action
        else:
            _log(f"[liar_dice_trajectories] max_turn={max_turn} reached (game {game_id})")
    except Exception as exc:
        _log(f"[liar_dice_trajectories] Failed to build episode (game {game_id}): {exc}")
        return []

    return examples
