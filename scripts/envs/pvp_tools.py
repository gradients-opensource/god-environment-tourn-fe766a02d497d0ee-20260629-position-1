"""Trimmed copy of the PvP tool-schema builders used by SFT generation.

Copied from /ephemeral/G.O.D core/pvp/tools.py (see §1.4 of
docs/SFT_ALIGNMENT_PLAN.md) — only the schema-generation functions needed to
build the `tools` field for training examples (4 memory tools + game_action),
matching exactly what LLMBot._run_turn exposes at eval time. `execute_memory_tool`
(eval-only dispatch against a live SlotMemory) is intentionally NOT copied.
"""

from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel

from envs.pvp_models import FunctionSchema
from envs.pvp_models import GameActionArgs
from envs.pvp_models import MemoryArea
from envs.pvp_models import MemoryConfig
from envs.pvp_models import MemoryOp
from envs.pvp_models import MemorySlotEdit
from envs.pvp_models import ToolSchema


GAME_ACTION_TOOL_NAME = "game_action"

# Presentation metadata — exactly one entry per enum member (asserted exhaustive).
_AREA_PURPOSE: dict[MemoryArea, str] = {
    MemoryArea.WORKING: "notes for THIS game, reset each game",
    MemoryArea.LONG_TERM: "notes on THIS opponent, persist across games",
}
_OP_PHRASING: dict[MemoryOp, tuple[str, str]] = {
    MemoryOp.REWRITE: ("Overwrite", "replaces the slot's previous content"),
    MemoryOp.APPEND: ("Append to", "oldest text drops if the slot is full"),
}
assert set(_AREA_PURPOSE) == set(MemoryArea), "every MemoryArea needs a purpose"
assert set(_OP_PHRASING) == set(MemoryOp), "every MemoryOp needs phrasing"


def memory_tool_name(area: MemoryArea, op: MemoryOp) -> str:
    return f"{area.value}_{op.value}"


# Reverse map for dispatch, generated from the same product the tools come from.
_TOOL_TO_AREA_OP: dict[str, tuple[MemoryArea, MemoryOp]] = {
    memory_tool_name(area, op): (area, op) for area in MemoryArea for op in MemoryOp
}


def _params_schema(model: type[BaseModel], *, slot_bounds: tuple[int, int] | None = None) -> dict:
    """JSON Schema for a tool's arguments, stripped of Pydantic titles.

    When slot_bounds is given, the integer 'slot' field carries minimum/maximum
    for the valid range. Advisory only (servers don't grammar-enforce tool args
    under tool_choice="auto"); SlotMemory rejects out-of-range slots regardless.
    """
    schema = deepcopy(model.model_json_schema())
    schema.pop("title", None)
    schema.pop("description", None)  # drop the model docstring; the function description covers it
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    if slot_bounds is not None and "slot" in schema.get("properties", {}):
        lo, hi = slot_bounds
        schema["properties"]["slot"]["minimum"] = lo
        schema["properties"]["slot"]["maximum"] = hi
    return schema


def _function_tool(name: str, description: str, parameters: dict) -> ToolSchema:
    return ToolSchema(function=FunctionSchema(name=name, description=description, parameters=parameters))


def build_memory_tools(configs: dict[MemoryArea, MemoryConfig]) -> list[ToolSchema]:
    """Generate memory tool schemas for the configured areas (one per area x op)."""
    out: list[ToolSchema] = []
    for area, cfg in configs.items():
        for op in MemoryOp:
            verb, effect = _OP_PHRASING[op]
            description = f"{verb} a {area.value} slot (slots 1-{cfg.n_slots}; {_AREA_PURPOSE[area]}); {effect}."
            parameters = _params_schema(MemorySlotEdit, slot_bounds=(1, cfg.n_slots))
            out.append(_function_tool(memory_tool_name(area, op), description, parameters))
    return out


def build_game_action_tool(legal_hint: str, legal_actions: list[int] | None = None) -> ToolSchema:
    """Schema for the turn-terminating move tool. legal_hint surfaces current legal ids.

    When legal_actions is given, action_id carries a JSON-Schema enum of the legal
    set. This is advisory: SGLang with tool_choice="auto" does not grammar-enforce
    argument values (verified empirically — malformed/illegal output reaches the
    wire), so the binding guard is the bot's validate-then-forfeit on the result.
    """
    parameters = _params_schema(GameActionArgs)
    if legal_actions is not None and "action_id" in parameters.get("properties", {}):
        parameters["properties"]["action_id"]["enum"] = list(legal_actions)
    return _function_tool(
        GAME_ACTION_TOOL_NAME,
        f"Commit your move and end your turn. {legal_hint}",
        parameters,
    )
