"""Trimmed copy of the PvP tool/memory pydantic models used by SFT generation.

Copied from /ephemeral/G.O.D core/models/pvp_models.py (see §1.4 of
docs/SFT_ALIGNMENT_PLAN.md) — only the models needed to build tool schemas and
memory configs for training data. Eval-only models (ChatMessage, ToolCall,
PvP*Result, etc.) and anything depending on core.constants.EnvironmentName are
intentionally NOT copied.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


# A single JSON scalar — the value type of tool-call arguments.
JsonScalar = str | int | float | bool | None


class FunctionSchema(BaseModel):
    """One function-tool exposed to the model. parameters is a JSON Schema document."""

    name: str
    description: str
    parameters: dict


class ToolSchema(BaseModel):
    """OpenAI function-tool envelope."""

    type: Literal["function"] = "function"
    function: FunctionSchema

    def to_openai(self) -> dict:
        return self.model_dump()


class MemoryArea(str, Enum):
    """An area of model-managed memory in the tool-calling harness.

    Add a member here (plus its SlotMemory instance and presentation metadata)
    to introduce a new memory area; the tool layer expands automatically.
    """

    WORKING = "working_memory"
    LONG_TERM = "long_term_memory"

    @property
    def persists_across_games(self) -> bool:
        """Long-term memory survives between games (opponent model); working resets."""
        return self is MemoryArea.LONG_TERM


class MemoryOp(str, Enum):
    """An edit operation on a memory slot.

    The value equals the SlotMemory method name, so dispatch needs no lookup
    table: getattr(slot_memory, op.value)(...).
    """

    REWRITE = "rewrite"
    APPEND = "append"


class MemoryConfig(BaseModel):
    """Sizing for one memory area: a fixed number of fixed-size slots."""

    n_slots: int = Field(gt=0, description="Number of addressable slots.")
    slot_token_budget: int = Field(gt=0, description="Max tokens retained per slot.")


class MemorySlotEdit(BaseModel):
    """Arguments accepted by a memory edit tool (rewrite/append)."""

    slot: int = Field(description="Target slot number.")
    content: str = Field(description="Text content for the slot.")


class GameActionArgs(BaseModel):
    """Arguments accepted by the game_action tool."""

    action_id: int = Field(description="A legal action id for the current state.")
