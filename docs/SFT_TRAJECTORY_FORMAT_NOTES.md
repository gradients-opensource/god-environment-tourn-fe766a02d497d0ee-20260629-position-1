# Env-server -> per-turn SFT example: translation flow

Notes for building a mock env server / generator test. Covers how a raw
`mcts-api` `/reset`+`/step` trajectory becomes the flattened
`{system, user, assistant, tools}` rows used for SFT (see
SFT_ALIGNMENT_PLAN.md §3 for the per-game generators).

## 1. Raw env-server response

`/reset` and `/step` both return:

```json
{"result": {"observation": "<raw_text>", "episode_id": "...", "reward": 0.0, "done": false, "info": {...}}}
```

`raw_text` shape (othello example):

```
# Game Rules
OTHELLO (REVERSI) RULES:
...
# Current Game State
Game: othello
You are Player 0.

Current State:
Black (x) to play:
  a b c d e f g h
1 - - - - - - - - 1
...
8 - - - - - - - - 8
  a b c d e f g h

Legal Actions:
  19 -> d3
  26 -> c4
  37 -> f5
  44 -> e6

Your choice (action ID only):
```

## 2. `_format_observation` (per-env, e.g. `othello_trajectories.py`,
`leduc_poker_env.py`)

Pure string transform, no network:
- Drop everything before `"Current State:"`.
- Split body at `"Legal Actions:"` -> `state_block` / `actions_block`.
- In `actions_block`, strip the 2-space indent (`"  19 -> d3"` -> `"19 -> d3"`)
  and rewrite `"Your choice (action ID only):"` -> `"Your choice (ID only):"`.
- Reassemble: `state_block + "\n" + "You are Player N." + "\n" + actions_block`.

Result (normalized envelope):

```
Current State:
Black (x) to play:
  a b c d e f g h
1 - - - - - - - - 1
...
  a b c d e f g h
You are Player 0.
Legal Actions:
19 -> d3
26 -> c4
37 -> f5
44 -> e6

Your choice (ID only):
```

## 3. `split_normalized_observation` (`pvp_format.py`, shared)

Regex-parses the envelope into `(state_desc, player_id, legal_actions)`:
1. Strip leading `"Current State:\n"`.
2. Split at the `"Legal Actions:"` header line -> `before` / `after`.
3. In `before`, find `"You are Player N."` -> `player_id`; remove that line ->
   `state_desc`.
4. In `after`, match each `"N -> label"` line -> `legal_actions = [(id, label), ...]`.

For the example above: `state_desc` = the board grid, `player_id = 0`,
`legal_actions = [(19,"d3"), (26,"c4"), (37,"f5"), (44,"e6")]`.

## 4. Per-env tweaks

- **Othello** (`othello_trajectories.py:139-140`): prepends
  `"You play x (Black).\n"` / `"You play o (White).\n"` to `state_desc` based
  on `player_id` (0=Black/x, 1=White/o).
- **Leduc Poker**: no extra tweak.

## 5. `_build_tool_example(observation, action_id)` -> one training row

```python
{
  "messages": [
    {"role": "system", "content": build_full_system_prompt(game_name)},
    {"role": "user", "content": build_user_prompt(state_desc, player_id, legal_actions)},
    {"role": "assistant", "content": None, "tool_calls": [
        {"type": "function", "function": {"name": "game_action", "arguments": json.dumps({"action_id": action_id})}}
    ]},
  ],
  "tools": json.dumps(tools_to_openai(build_pvp_tools([id for id, _ in legal_actions]))),
}
```

- `system` = rules (from `pvp_assets/pvp_game_prompts.yml`) + empty memory
  block (`EMPTY_MEMORY_BLOCK`) + `TOOL_GUIDANCE`. Constant per game, computed
  once as `_SYSTEM_PROMPT`.
- `user` = `f"Current state:\n{state_desc}\n\nYou are Player {player_id}.\nLegal actions:\n{id} -> {label}\n..."`.
- `tools` = 4 memory tools (`working_memory_rewrite/append`,
  `long_term_memory_rewrite/append`) + `game_action`
  (`{action_id: int}`, `enum=<legal_action_ids>`).
- `action_id` = whichever id the per-env policy (heuristic / random / expert)
  picked from `legal_actions` for this turn.

## 6. Episode loop

For each turn: build the example from the current observation (steps 2-5) ->
policy picks `action_id` -> `POST /step {"action": str(action_id), "episode_id": ...}`
-> new raw observation -> repeat until `done` or `max_turn`. Returns
`(examples, final_reward)`, where `final_reward` is the env's terminal score
already normalized to `[0, 1]` (0.0 = loss, 0.5 = draw, 1.0 = win), used by
`generate_trajectories.py` for score-based sampling.

## Mocking

The only seam that needs network I/O is step 1 (`/reset`/`/step`). Everything
from step 2 onward is deterministic pure-Python string/regex transforms. A
mock server that returns canned `raw_text` fixtures (one "reset" + N "step"
responses per game, in the format shown in §1) is enough to exercise the full
pipeline through `_build_tool_example` without a live `mcts-api` sidecar.
