"""Tool-calling prompt/tool constants for InterCode — copied from G.O.D's
validator/evaluation/eval_intercode.py (see §1.4/§2.2 of docs/SFT_ALIGNMENT_PLAN.md).

Keeping these in sync with the evaluator ensures training data matches the
exact prompt structure / tool schemas the model sees at eval time
(``_build_tool_messages`` / ``build_intercode_action_tools`` /
``_format_tool_history``).
"""

from envs.pvp_models import FunctionSchema
from envs.pvp_models import ToolSchema

DEFAULT_OBS_TRUNCATE_CHARS = 350
DEFAULT_MAX_TURNS = 10

INTERCODE_EXECUTE_TOOL_NAME = "execute_bash"
INTERCODE_SUBMIT_TOOL_NAME = "submit"

INTERCODE_TOOL_SYSTEM_PROMPT = """Interact with a Bourne Shell system using Bash commands to answer a question.
You may reason briefly in normal assistant text, but every turn must call exactly one tool:
(1) execute_bash, which executes Bash code on the Bourne Shell system
(2) submit, which indicates that the previous observation is the answer
Do not write actions as plain text. Use the tools for every action.

Examples:

Question: Convert the first 16 characters in "/testbed/textfile7.txt" to a single hexadecimal value
Turn 1 tool: execute_bash(command="xxd -p -l 16 /testbed/textfile7.txt")
Observation 1: /bin/bash: line 1: xxd: command not found
Turn 2 tool: execute_bash(command="od -A n -t x1 -N 16 /testbed/textfile7.txt")
Observation 2:  54 65 78 74 20 66 69 6c 65 20 37 20 77 69 74 68
Turn 3 tool: execute_bash(command="od -A n -t x1 -N 16 /testbed/textfile7.txt | awk '{$1=$1;print}' | tr -d ' '")
Observation 3: 546578742066696c6520372077697468
Turn 4 tool: submit()

Question: print disk usage in human readable format of files or folders in /workspace
Turn 1 tool: execute_bash(command="du /workspace")
Observation 1: 48\t/workspace/dir1
8\t/workspace/dir2/mysql
24\t/workspace/dir2
100\t/workspace
Turn 2 tool: execute_bash(command="du -h /workspace")
Observation 2: 48K\t/workspace/dir1
8.0K\t/workspace/dir2/mysql
24K\t/workspace/dir2
100K\t/workspace
Turn 3 tool: execute_bash(command="du -sh /workspace")
Observation 3: 100K\t/workspace
Turn 4 tool: submit()

Question: Count all the lines of all php files in the /testbed directory recursively
Turn 1 tool: execute_bash(command="find /testbed -name \"*.php\" | xargs wc -l")
Observation 1:  1 /testbed/dir1/info.php
 1 /testbed/hello.php
 2 total
Turn 2 tool: execute_bash(command="find /testbed -name \"*.php\" -type f -exec cat {} + | wc -l")
Observation 2: 2
Turn 3 tool: submit()
"""


def _function_tool(name: str, description: str, parameters: dict) -> ToolSchema:
    return ToolSchema(function=FunctionSchema(name=name, description=description, parameters=parameters))


def build_intercode_action_tools() -> list[ToolSchema]:
    return [
        _function_tool(
            INTERCODE_EXECUTE_TOOL_NAME,
            "Execute one Bash command in the InterCode Bourne Shell environment.",
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The Bash command to execute.",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        ),
        _function_tool(
            INTERCODE_SUBMIT_TOOL_NAME,
            "Submit the previous observation as the final answer and end the task.",
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        ),
    ]


def _format_tool_history(history: list[str]) -> str:
    if not history:
        return "No commands have been executed yet."
    return "\n".join(history)


def build_user_prompt(query: str, history: list[str], turn: int) -> str:
    """Mirrors eval_intercode._build_tool_messages's user-message content."""
    return (
        f"Question: {query}\n\n"
        f"Previous steps:\n{_format_tool_history(history)}\n\n"
        f"Turn {turn}: call exactly one tool now."
    )
