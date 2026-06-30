"""PvP system prompt builder — single source of truth for all env games.

Loads core/config/pvp_game_prompts.yml (canonical copy from G.O.D-game) so
that SFT trajectory data is trained with the exact same system prompt the
validator uses at PvP eval time.

Also carries the prompt-snippet functions copied from /ephemeral/G.O.D
core/pvp/bot.py (see §1.4 of docs/SFT_ALIGNMENT_PLAN.md): the per-turn
system/user prompts are exactly
    generate_system_prompt() + memory_block() + TOOL_GUIDANCE   (system)
    "Current state:\n{state}\n\nYou are Player {p}.\nLegal actions:\n{lines}"  (user)
so SFT examples match LLMBot._run_turn's stateless per-turn reconstruction
byte-for-byte (rules text + an always-empty memory block + tool guidance).

Usage:
    from envs.pvp_format import SYSTEM_PROMPT_LIARS_DICE
    from envs.pvp_format import SYSTEM_PROMPT_GIN_RUMMY
    from envs.pvp_format import SYSTEM_PROMPT_LEDUC_POKER
    from envs.pvp_format import build_full_system_prompt, build_user_prompt, build_pvp_tools
"""

import re
from pathlib import Path

import yaml

from envs.pvp_constants import PVP_LONGTERM_MEM_SLOTS
from envs.pvp_constants import PVP_LONGTERM_SLOT_TOKENS
from envs.pvp_constants import PVP_WORKING_MEM_SLOTS
from envs.pvp_constants import PVP_WORKING_SLOT_TOKENS
from envs.pvp_memory import SlotMemory
from envs.pvp_memory import WhitespaceTokenCounter
from envs.pvp_models import MemoryArea
from envs.pvp_models import MemoryConfig
from envs.pvp_models import ToolSchema
from envs.pvp_tools import build_game_action_tool
from envs.pvp_tools import build_memory_tools

_PROMPTS_PATH = Path(__file__).resolve().parent / "pvp_assets" / "pvp_game_prompts.yml"


def _load_prompts() -> dict[str, str]:
    with open(_PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def build_system_prompt(game_name: str) -> str:
    """Old plain-text format ("respond with ONLY the action ID").

    Still used by the not-yet-restructured GRPO envs (gin_rummy_env.py,
    leduc_poker_env.py) and liar_dice_trajectories.py's SFT generator. Use
    build_tool_system_prompt / build_full_system_prompt for the new
    tool-calling format (§3.2/§3.3).
    """
    prompts = _load_prompts()
    rules_key = f"{game_name}_rules"
    if rules_key not in prompts:
        raise ValueError(f"Unknown game: {game_name!r} (no {rules_key} in pvp_game_prompts.yml)")
    return prompts["system_prompt_template"].format(game_name=game_name, rules=prompts[rules_key])


def build_tool_system_prompt(game_name: str) -> str:
    """New tool-calling format's rules block — matches G.O.D's pvp_game_prompts.yml
    system_prompt_template (rules only, no "Output Format" section). The memory
    block + tool guidance are appended separately (see build_full_system_prompt)."""
    prompts = _load_prompts()
    rules_key = f"{game_name}_rules"
    if rules_key not in prompts:
        raise ValueError(f"Unknown game: {game_name!r} (no {rules_key} in pvp_game_prompts.yml)")
    return prompts["tool_system_prompt_template"].format(game_name=game_name, rules=prompts[rules_key])


SYSTEM_PROMPT_LIARS_DICE  = build_system_prompt("liars_dice")
SYSTEM_PROMPT_GIN_RUMMY   = build_system_prompt("gin_rummy")
SYSTEM_PROMPT_LEDUC_POKER = build_system_prompt("leduc_poker")


# ---------------------------------------------------------------------------
# Tool guidance text — verbatim from core/pvp/bot.py
# ---------------------------------------------------------------------------

TOOL_GUIDANCE = (
    "You get ONE response this turn. In it, optionally edit your memory notes, and "
    "then call game_action with a legal action id to commit your move. If you do not "
    "call game_action, you forfeit the turn — so always include it."
)
REFLECTION_GUIDANCE = (
    "The game is over. Use the memory tools to update your long-term notes on this "
    "opponent for future games — keep durable, generalisable reads (their tendencies, "
    "your counter-strategy) and drop move-by-move detail. There is no move to make."
)


# ---------------------------------------------------------------------------
# Memory — default areas, render-to-prompt, and the empty memory block
# ---------------------------------------------------------------------------

def default_memories() -> dict[MemoryArea, SlotMemory]:
    """Build the standard working + long-term memory areas from constants.

    Mirrors core/pvp/bot.py::default_memories. Training always starts from
    these freshly-reset (all-empty) areas — SFT phase 1 never simulates a
    matchup, so the memory block is always the "all (empty)" render.
    """
    counter = WhitespaceTokenCounter()
    return {
        MemoryArea.WORKING: SlotMemory(PVP_WORKING_MEM_SLOTS, PVP_WORKING_SLOT_TOKENS, counter),
        MemoryArea.LONG_TERM: SlotMemory(PVP_LONGTERM_MEM_SLOTS, PVP_LONGTERM_SLOT_TOKENS, counter),
    }


def memory_block(memories: dict[MemoryArea, SlotMemory]) -> str:
    """Render the WORKING_MEMORY / LONG_TERM_MEMORY block. Mirrors LLMBot._memory_block."""
    return "\n\n".join(
        mem.render(title=f"{area.value.upper()} (your notes):") for area, mem in memories.items()
    )


EMPTY_MEMORY_BLOCK = memory_block(default_memories())


# ---------------------------------------------------------------------------
# Per-turn system / user prompts — mirrors LLMBot._system_prompt / _user_prompt
# ---------------------------------------------------------------------------

def build_full_system_prompt(game_name: str, memories: "dict[MemoryArea, SlotMemory] | None" = None) -> str:
    """rules + memory block + tool guidance. Mirrors LLMBot._system_prompt.

    memories defaults to a fresh (all-empty) set — SFT phase 1 always trains
    on the empty memory block, since the generator never emits memory edits.
    """
    block = memory_block(memories) if memories is not None else EMPTY_MEMORY_BLOCK
    return "\n\n".join([build_tool_system_prompt(game_name), block, TOOL_GUIDANCE])


def build_reflection_system_prompt(game_name: str, memories: "dict[MemoryArea, SlotMemory] | None" = None) -> str:
    """rules + memory block + reflection guidance. Mirrors LLMBot._reflection_system_prompt."""
    block = memory_block(memories) if memories is not None else EMPTY_MEMORY_BLOCK
    return "\n\n".join([build_tool_system_prompt(game_name), block, REFLECTION_GUIDANCE])


def legal_hint(legal_actions: list[int]) -> str:
    """Mirrors LLMBot._legal_hint — surfaces the legal action ids to game_action's schema."""
    return "Legal action ids: " + ", ".join(str(a) for a in legal_actions) + "."


def action_line(action_id: int, label: str) -> str:
    """Mirrors LLMBot._action_line's output shape ("{id} -> {label}").

    At eval time this comes from state.action_to_string(); training-side, the
    env server's observation text already provides "{id} -> {label}" lines
    (see e.g. liar_dice_env.py's parse_game_state), so callers pass those
    labels through directly rather than re-deriving them.
    """
    return f"{action_id} -> {label}"


def build_user_prompt(state_desc: str, player_id: int, legal_actions: list[tuple[int, str]]) -> str:
    """Mirrors LLMBot._user_prompt.

    legal_actions is a list of (action_id, label) pairs, in the order they
    should be presented (matches the order action_line/_action_line would
    produce for state.legal_actions(player_id)).
    """
    action_lines = "\n".join(action_line(aid, label) for aid, label in legal_actions)
    return (
        f"Current state:\n{state_desc}\n\n"
        f"You are Player {player_id}.\n"
        f"Legal actions:\n{action_lines}"
    )


# ---------------------------------------------------------------------------
# Normalized observation -> (state_desc, player_id, legal_actions)
# ---------------------------------------------------------------------------
#
# Several env servers' raw observations are normalized (by
# gin_rummy_env.extract_and_format_observation / leduc_poker_env._format_observation /
# liar_dice_trajectories._split_observation's input) into a common
#     "Current State:\n<state text>You are Player N.\nLegal Actions:\n<id> -> <label>\n..."
# envelope. split_normalized_observation extracts the (state_desc, player_id,
# legal_actions) pieces build_user_prompt needs to reproduce the eval-time user
# prompt for that turn.

_RE_CURRENT_STATE_PREFIX = re.compile(r"^Current State:\n", re.MULTILINE)
_RE_LEGAL_ACTIONS_HEADER = re.compile(r"^[ \t]*Legal Actions:[ \t]*$", re.MULTILINE)
_RE_PLAYER_LINE = re.compile(r"^You are Player (\d+)\.[ \t]*$", re.MULTILINE)
_RE_LEGAL_ACTION_LINE = re.compile(r"^[ \t]*(\d+)[ \t]*->[ \t]*(.+?)[ \t]*$")


def split_normalized_observation(observation: str) -> "tuple[str, int, list[tuple[int, str]]]":
    """Split a normalized observation into (state_desc, player_id, legal_actions)."""
    text = observation
    prefix = _RE_CURRENT_STATE_PREFIX.search(text)
    if prefix:
        text = text[prefix.end():]

    header = _RE_LEGAL_ACTIONS_HEADER.search(text)
    if not header:
        raise ValueError(f"Could not find 'Legal Actions:' header in observation: {observation[:200]!r}")
    before, after = text[:header.start()], text[header.end():]

    player_match = _RE_PLAYER_LINE.search(before)
    if not player_match:
        raise ValueError(f"Could not find 'You are Player N.' line in observation: {observation[:200]!r}")
    player_id = int(player_match.group(1))
    before = _RE_PLAYER_LINE.sub("", before)
    state_desc = before.strip("\n")

    legal_actions: "list[tuple[int, str]]" = []
    for line in after.splitlines():
        m = _RE_LEGAL_ACTION_LINE.match(line)
        if m:
            legal_actions.append((int(m.group(1)), m.group(2)))
        elif legal_actions:
            break
    if not legal_actions:
        raise ValueError(f"Could not parse legal actions from observation: {observation[:200]!r}")

    return state_desc, player_id, legal_actions


def build_reflection_user_prompt(state_desc: str, outcome: str) -> str:
    """Mirrors LLMBot._reflection_user_prompt. outcome is GameOutcome.value ("win"/"loss"/"draw")."""
    return (
        f"The game is over. Result for you: {outcome.upper()}.\n\n"
        f"Final state:\n{state_desc}\n\n"
        "Update your long-term notes on this opponent for future games."
    )


# ---------------------------------------------------------------------------
# Standard per-turn tools list — always all 5 (4 memory tools + game_action),
# per §3.2/§3.3a
# ---------------------------------------------------------------------------

def build_pvp_tools(legal_actions: list[int]) -> list[ToolSchema]:
    """4 memory tools (empty-config) + game_action with enum=legal_actions.

    Mirrors LLMBot.__init__'s self._memory_tools + _run_turn's
    [..., build_game_action_tool(...)] — the exact tools list shape the model
    sees at eval time, regardless of whether the generator ever emits memory
    tool calls (it doesn't, in phase 1 — see §3.2).
    """
    memories = default_memories()
    memory_tools = build_memory_tools(
        {area: MemoryConfig(n_slots=mem.n_slots, slot_token_budget=mem.slot_token_budget)
         for area, mem in memories.items()}
    )
    return memory_tools + [build_game_action_tool(legal_hint(legal_actions), legal_actions)]


def tools_to_openai(tools: list[ToolSchema]) -> list[dict]:
    """Convert ToolSchema list to plain dicts for the `tools` field of a training row."""
    return [t.to_openai() for t in tools]
