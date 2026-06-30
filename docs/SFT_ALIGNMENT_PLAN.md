# SFT Alignment Plan — bringing this repo's env training in line with G.O.D's PvP rewrite

This is a re-creation of a plan doc that existed locally but was never
committed. It tracks the work needed to make this repo's environment-task
training (`scripts/envs/*`) match the tool-calling + memory PvP eval harness
introduced in `/ephemeral/G.O.D` (see that repo's
`docs/MINER_PVP_TOURNAMENT_GUIDE.md` and `docs/TRAINING_FORMAT_ALIGNMENT.md`).

Referenced as `docs/SFT_ALIGNMENT_PLAN.md §x.y` from comments across
`scripts/envs/`. Keep section numbers stable — code comments point at them.

---

## 0. Status checklist (review/update this)

Reconstructed from git history (commits up to `e1ef2cc`) + this session's
work. Edit freely — this is the live tracker.

- [x] **§1 Shared groundwork (Phase 0)** — `pvp_models.py`/`pvp_tools.py`/
      `pvp_memory.py`/`pvp_format.py`/`pvp_constants.py`/
      `pvp_assets/pvp_game_prompts.yml` copied from G.O.D. Commit `66d0787`.
- [x] **§2 InterCode SFT alignment** — `intercode_format.py` +
      `intercode_dataset.py`. Commits `14447d2`/`4668d44`/`b7f67f3`.
- [x] **§3.1 Liar's Dice SFT generator** — `liar_dice_trajectories.py`
      (MCTS-probability expert). Commit `e1ef2cc`.
- [x] **§3.2 Gin Rummy SFT generator** — `gin_rummy_trajectories.py`
      (hand-coded deadwood/meld heuristic). Commit `e1ef2cc`.
- [x] **§3.3 Leduc Poker SFT generator** — `leduc_poker_trajectories.py`
      (random policy + score sampling). Commit `e1ef2cc`.
- [x] **§3.4 Othello SFT generator** — code-complete AND verified live
      against a real `mcts-api` env server (2026-06-13 session); not yet
      committed.
      Added `othello_trajectories.py` (`generate_heuristic_episode`,
      corner/edge heuristic), registered in `sft_env_configs.py`
      (`_SFT_REGISTRY["othello"]`) and `generate_trajectories.py`
      (`_FLAT_EXAMPLE_ENVS`, `_ENV_DEFAULT_ARGS["othello"]` = 4000 games /
      max_turn 70 / mcts_sims 25 / sample_by_score).
      - Verified against `/ephemeral/G.O.D/pvp_smoke_all.json` (real eval
        trajectory): system prompt, user prompt (incl. "You play x/o" colour
        prefix + Black=Player0/White=Player1 mapping), `tools` (all 5,
        `game_action.enum` = legal ids), single-`game_action`/`content: null`
        assistant turn — all match (§3.1/3.2 satisfied). Action labels
        confirmed to be 2-char cell refs (`f2`, `f3`, ...), matching the
        corner/edge heuristic's regex.
      - **Live-verified (§4, see below)**: ran against a real
        `gradientsio/mcts-api:latest` sidecar — `_format_observation` parses
        the actual raw `/reset`/`/step` observation correctly, won games
        produce correct `(examples, final_reward=1.0)`, output rows match
        `pvp_smoke_all.json` byte-for-byte in shape. The "one unverifiable
        assumption" from the previous write-up is now confirmed correct.
- [x] **§4 Validation against real harness** — live env-server validation
      done for all 4 game envs + intercode (2026-06-13 session); model-level
      validation (`pvp_smoke_match.py` on a trained checkpoint) still
      outstanding. See §4 for details and the 3 bugs found/fixed.
- [x] **§5.1 Memory-tool training — Liar's Dice only** — done (2026-06-17
      session, not yet committed). `liar_dice_policy.choose_action`'s
      opponent-aware posterior (commit `63799d4`) depends on this hand's full
      bid history, but the visible state text only ever shows the most recent
      bid — so the label wasn't a function of anything in the row. Fixed by
      having each turn write a one-line bid-history note via
      `working_memory_append` alongside `game_action`, with the next turn's
      system prompt rebuilt from the accumulated notes. See §5.1 below for
      detail. Gin Rummy/Leduc/Othello are still fully stateless (no
      history-dependent policy, so no gap to close); the long_term-memory /
      cross-game reflection half of this stretch goal remains unstarted.
- [ ] **§5.2 GRPO-side tool-calling rollouts** — not started, stretch. GRPO
      envs (`gin_rummy_env.py`, `leduc_poker_env.py`, `liar_dice_env.py`,
      etc.) still use the old `Thought:/Action:<id>` format — only SFT has
      been migrated so far.
- [ ] **§5.3 Othello GRPO env** — not started, stretch. No `othello_env.py`
      GRPO rollout exists.

---

## 1. Shared groundwork ("Phase 0") — DONE

Commit `66d0787` ("SFT groundwork for tool-calling eval format (Phase 0)").

Copies the eval-side building blocks from `/ephemeral/G.O.D` into
`scripts/envs/` so SFT trajectory generation can reproduce the exact eval-time
prompts/tools:

- **§1.1** `pvp_models.py` — trimmed copy of `core/models/pvp_models.py`
  (`ToolSchema`, `FunctionSchema`, `MemoryArea`, `MemoryOp`, `MemoryConfig`,
  `MemorySlotEdit`, `GameActionArgs`). Eval-only models not copied.
- **§1.2** `pvp_tools.py` — trimmed copy of `core/pvp/tools.py`
  (`build_memory_tools`, `build_game_action_tool`, schema helpers).
  `execute_memory_tool` (eval-only dispatch) not copied.
- **§1.3** `pvp_memory.py` — verbatim copy of `core/pvp/memory.py`
  (`SlotMemory`, `WhitespaceTokenCounter`) so the rendered memory block is
  byte-for-byte what `LLMBot` renders.
- **§1.4** `pvp_format.py` + `pvp_constants.py` + `pvp_assets/pvp_game_prompts.yml`
  — system/user prompt builders mirroring `core/pvp/bot.py`
  (`build_full_system_prompt`, `build_reflection_system_prompt`,
  `build_user_prompt`, `build_pvp_tools`, `memory_block`, `TOOL_GUIDANCE`,
  `REFLECTION_GUIDANCE`, `split_normalized_observation`) plus the canonical
  rules text copied from `core/config/pvp_game_prompts.yml`
  (`liars_dice_rules`, `leduc_poker_rules`, `gin_rummy_rules`,
  `othello_rules`, `system_prompt_template`, `tool_system_prompt_template`).

Status: **done**. Constants (`PVP_WORKING_MEM_SLOTS=4`,
`PVP_LONGTERM_MEM_SLOTS=8`, 128 tokens/slot, etc.) match
`core/pvp/constants.py`.

---

## 2. InterCode alignment — DONE

Commits `14447d2`, `4668d44`, `b7f67f3` ("intercode", "update intercode",
"intercode sft update").

- **§2.1** `intercode_format.py` — tool-calling prompt/tool constants copied
  from `validator/evaluation/eval_intercode.py`
  (`INTERCODE_TOOL_SYSTEM_PROMPT`, `build_intercode_action_tools`,
  `execute_bash`/`submit` schemas, `_format_tool_history`-equivalent).
- **§2.2** `intercode_dataset.py` — builds the SFT dataset from the
  validator-mounted miner dataset
  (`gradients-io-tournaments/intercode_bigcode_combined_12k`), one row per
  example shaped `{system, user, assistant(tool_calls=[execute_bash])}` +
  `tools`, matching `eval_intercode._build_tool_messages` turn-1 shape.
- `train_sft_env.py` wired to call `build_intercode_sft_dataset()` via the
  offline-env path (`_OFFLINE_ENVS = {"intercode"}` in
  `sft_env_configs.py`/`generate_trajectories.py`).

Status: **done**.

---

## 3. Per-game PvP SFT trajectory generators

Goal (from `TRAINING_FORMAT_ALIGNMENT.md` §3.1/3.2): each game's SFT
generator emits **flattened, stateless, per-turn** examples —
`{"messages": [system, user, assistant(tool_calls=[game_action])], "tools": [...]}` —
where `system` = `build_full_system_prompt(game)` (rules + always-empty memory
block + `TOOL_GUIDANCE`), `user` = `build_user_prompt(state_desc, player_id,
legal_actions)`, and `tools` = `build_pvp_tools(legal_action_ids)` (4 memory
tools + `game_action`, enum-constrained). This is **§3.1 (required)**; §3.2
(statelessness) falls out for free because each turn is its own row; §3.3
(memory tools in training) is **not** attempted — memory tools are offered but
never exercised by the assistant turn, same as the eval-time "ignore memory,
just play" baseline.

### 3.1 Liar's Dice — DONE (commit `e1ef2cc`)

`liar_dice_trajectories.py::generate_expert_episode`:
- Plays against the `mcts` opponent (`mcts_max_simulations=225`).
- `get_expert_action` does probability-weighted sampling over
  `parse_game_state(...).actions` (softmax, `_SAMPLING_TEMPERATURE=0.01` ≈
  near-greedy on the MCTS policy's action probabilities).
- `_split_observation` parses the env server's raw
  `"Current State:\n...\nYou are Player N.\nLegal Actions:\n..."` envelope
  into `(state_desc, player_id, legal_actions)`.
- `_build_tool_example` builds the flattened row via
  `build_full_system_prompt("liars_dice")` + `build_user_prompt` +
  `build_pvp_tools`.
- Registered in `sft_env_configs.py` (`_SFT_REGISTRY["liars_dice"]`) and
  `generate_trajectories.py` (`_FLAT_EXAMPLE_ENVS`,
  `_ENV_DEFAULT_ARGS["liars_dice"]` = 50000 games / max_turn 30 /
  mcts_sims 225).

### 3.2 Gin Rummy — DONE (commit `e1ef2cc`)

`gin_rummy_trajectories.py::generate_expert_episode`:
- Plays against `mcts` (`mcts_max_simulations` randomized 25-50).
- `get_expert_action` is a hand-coded heuristic (deadwood/meld/knock logic
  copied & adapted from the old `gin_rummy_env.py` GRPO heuristics): phase
  dispatch over Draw/Discard/Knock/Layoff using
  `compute_optimal_deadwood`/`get_optimal_meld_cards`/`choose_discard`/
  `choose_draw`/`choose_meld_or_layoff_action`.
- `extract_and_format_observation` (existing, from `gin_rummy_env.py`)
  normalizes the raw observation; `split_normalized_observation` then splits
  it into `(state_desc, player_id, legal_actions)`.
- Same flattened-row shape via `_build_tool_example`.
- Registered: `_SFT_REGISTRY["gin_rummy"]`, `_FLAT_EXAMPLE_ENVS`,
  `_ENV_DEFAULT_ARGS["gin_rummy"]` = 4000 games / max_turn 200 /
  mcts_sims 25.

### 3.3 Leduc Poker — DONE (commit `e1ef2cc`)

`leduc_poker_trajectories.py::generate_random_episode`:
- Plays against `mcts` (`mcts_max_simulations=50`) with a **uniformly random**
  policy (`_random_action`) — Leduc's tiny state space means MCTS plays
  near-optimally, so a hand-coded expert rarely beats it; random + score
  filtering is more practical.
- Returns `(examples, final_reward)` so `generate_trajectories.py` can apply
  `--wins-only` / `--sample-by-score`.
- `_format_observation` (existing, from `leduc_poker_env.py`) normalizes the
  raw observation; `split_normalized_observation` splits it.
- Registered: `_SFT_REGISTRY["leduc_poker"]`, `_FLAT_EXAMPLE_ENVS`,
  `_ENV_DEFAULT_ARGS["leduc_poker"]` = 200000 games / max_turn 10 /
  mcts_sims 50, `sample_by_score=True`, `score_power=3.0`.

### 3.4 Othello — DONE (not yet committed)

Added `othello_trajectories.py`, registered in `sft_env_configs.py`
(`_SFT_REGISTRY["othello"]`) and `generate_trajectories.py`
(`_FLAT_EXAMPLE_ENVS`, `_ENV_DEFAULT_ARGS["othello"]` = 4000 games /
max_turn 70 / mcts_sims 25, `sample_by_score=True`, `score_power=2.0`).

- A trajectory generator (`generate_heuristic_episode`) playing against
  `mcts`, producing flattened `{system, user, assistant(tool_calls=[game_action])}`
  rows via `build_full_system_prompt("othello")` / `build_user_prompt` /
  `build_pvp_tools`, exactly mirroring §3.1-3.3's shape.
- **Othello-specific quirk** (G.O.D's `core/pvp/agents.py`
  `OthelloAgent.format_state`): the eval-time state description is prefixed
  with `"You play x (Black)."` / `"You play o (White)."` based on `player_id`
  (0=Black/x, 1=White/o). `_format_observation` reproduces this prefix on
  `state_desc` before calling `build_user_prompt` — **live-verified** against
  a real `mcts-api` sidecar to match `pvp_smoke_all.json`.
- Move-selection policy: a corner/edge-aware heuristic
  (`get_heuristic_action`/`_cell_score`/`_softmax_weights`) — prefer corners,
  avoid X-squares adjacent to empty corners, mild preference for edges,
  otherwise weighted-random over 2-char cell labels (`d3`, `f2`, ...).
- **Live validation (2026-06-13 session)**: ran
  `generate_heuristic_episode(task_id=412345678, ...)` (task_id inside
  Othello's `GAMES_TO_TASK_ID_RANGE` `(400000000, 499999999)`) against a real
  `gradientsio/mcts-api:latest` sidecar — won the game (`final_reward=1.0`),
  produced 32 examples, and the board rendering / `"You play x (Black)."` /
  `"Black (x) to play:"` grid / `"You are Player 0."` / legal-move labels
  matched `pvp_smoke_all.json` exactly. A second run with a small
  `--max_turn` and a losing/drawing game correctly produced 0 examples
  (score-filtered) — confirms `sample_by_score`/`score_power=2.0` behaves as
  intended (only wins are kept; see §6.3 for the general sampling note).

---

## 4. Validation — partially done

Per `TRAINING_FORMAT_ALIGNMENT.md` §4:

1. **DONE (2026-06-13 session)** — ran `generate_trajectories.py` for each
   of the 4 game envs (liars_dice, gin_rummy, leduc_poker, othello)
   individually against live `mcts-api` sidecars on `internal_bridge`
   (172.18.0.2-5), plus `intercode` against the real mounted miner dataset
   (`gradients-io-tournaments--intercode_bigcode_combined_12k`, 12000 rows →
   5996 step-examples). Spot-checked rows against `pvp_smoke_all.json`:
   `tools` lists match, each `tool_calls` has exactly one entry
   (`game_action` or `execute_bash`) with a legal `action_id`/correct
   `command`, `content` is `null`. Also ran the **production multi-env path**
   (`generate_trajectories.py`'s multi-env branch + `merge_datasets`) with all
   5 envs together → 10038-row merged dataset; see §6 for the bug this
   surfaced and its fix.
   - Also ran `tokenize_and_mask` end-to-end with the real
     `Qwen/Qwen2.5-3B-Instruct` tokenizer on 10 sampled rows spanning all 5
     envs — correct `<tool_call>{...}</tool_call>` rendering and correct
     `assistant_masks` (assistant-turn-only loss, see §6.4) for both
     `game_action` and `execute_bash` tool types.
2. **STILL TODO (needs GPU + a completed training run)** — run the
   fine-tuned checkpoint through `/ephemeral/G.O.D/scripts/pvp_smoke_match.py`
   and compare forfeit rates before/after this restructuring. This is the
   only remaining item blocking full confidence in the new format; everything
   upstream of it (data generation, merging, tokenization, masking) has now
   been live-validated.

---

## 5. Stretch / advanced ideas

Captured from earlier discussion — revisit only after §3.4/§4 land:

### §5.1a Working-memory bid-history training (Liar's Dice) — DONE (2026-06-17 session, not yet committed)

**The flaw this fixes**: `liar_dice_policy.choose_action` (commit `63799d4`)
scores legal bids against a Bayesian posterior over the opponent's hand built
from `_state["observations"]` — the *full* sequence of opponent bids observed
so far this hand, tracked across calls via module-level state. But the
visible `state_desc` shown to the model only ever contains the single most
recent bid (`_RE_BID` only matches `"Current bid:"` — and the real eval-time
text is identical: `core/pvp/agents.py::LiarsDiceAgent.format_state` keeps
only `bid_parts[-1]`, discarding everything earlier). Since the SFT row's
input never carried that history and the assistant turn never touched memory,
the label was a function of information invisible to both the training row
*and* the real eval-time model (which rebuilds its prompt from scratch every
turn — `MINER_PVP_TOURNAMENT_GUIDE.md` §4 — and only ever sees what's written
to its own memory slots). Two states with identical visible `(dice,
current_bid)` but different hidden histories could get different "correct"
labels from the expert — aliased, contradictory supervision.

**Fix**: `liar_dice_policy.choose_action` now also returns a one-line
`memory_note` (e.g. `"opponent bid 3-4; I bid 4-4."` / `"no prior bid; I
called Liar."`) describing the round it just resolved.
`liar_dice_trajectories.generate_expert_episode` keeps a per-episode
`SlotMemory` (via `pvp_format.default_memories()`), rebuilds the system
prompt every turn from its current contents (instead of the constant
always-empty block other envs use), and has each turn's assistant message
call `working_memory_append` (slot 1) with that note *before* `game_action`
— mirroring `core/pvp/bot.py::LLMBot._run_turn`'s "any number of memory tool
calls plus exactly one game_action call" contract exactly. `choose_action`'s
cross-turn `_state` is now reset via an explicit `reset_episode()` call at
the start of each hand (previously reset only on "observed dice changed",
which could silently leak a finished hand's bid history into an unrelated
new hand that happened to deal identical dice in the same worker process).

**Verified live** (2026-06-17, real `mcts-api` sidecar via
`examples/run_env_sidecars.sh`/`run_environment_task.sh`'s approach): ran
several games, confirmed turn 0's system prompt has the all-empty memory
block (matches the old behavior for turn 0), and each subsequent turn's
`WORKING_MEMORY` block in the system prompt contains the prior turns' notes
verbatim, e.g. by turn 2 of a 3-turn game: `"[1] no prior bid; I bid 2-1.\nopponent
bid 3-2; I bid 3-3."` — the label-relevant history is now actually present in
the row that's labeled.

**Scope note**: this only covers *working* memory within a single hand.
Gin Rummy/Leduc Poker/Othello's generators have no analogous gap (their
policies don't depend on cross-turn history hidden from the visible state),
so they're untouched. The cross-game `long_term_memory_*` / reflection piece
below remains unstarted.

### §5.1b Long-term-memory / reflection training — not started, stretch

Extend generators to simulate a multi-game matchup against the same opponent
so `long_term_memory_*` tools can be exercised across games, plus a
reflection-turn example (`build_reflection_system_prompt` +
`build_reflection_user_prompt`, memory-tools-only). Real engineering work;
only matters if competing on the memory/reflection axis.
- **§5.2 GRPO-side tool-calling rollout** — `env_configs.py`'s GRPO rollouts
  (`gin_rummy_env.py`, `leduc_poker_env.py`, `liar_dice_env.py`,
  `goof_spiel_env.py`, etc.) still use the old
  `"Thought:/Action:<id>"` plain-text format (Option 2 in
  `TRAINING_FORMAT_ALIGNMENT.md` §3.1 is currently SFT-only). Switching GRPO
  rollouts to tool-call format + `apply_chat_template(..., tools=...)` would
  align RL training the same way SFT now does, but is a bigger change
  (tokenizer-template-dependent prompt/parsing for every env).
- **§5.3 Othello GRPO env** — no `othello_env.py` GRPO rollout/curriculum
  exists at all (only SFT per §3.4). Add one if Othello should also be RL-
  trained, not just SFT-warm-started.

---

## 6. Bugs found & fixed during live stress testing (2026-06-13 session)

After §3.4 (Othello) was wired up, a full live stress test was run: all 4
game generators against real `mcts-api` sidecars on `internal_bridge`, plus
intercode against the real mounted miner dataset, individually and through
the production multi-env path. This surfaced two pre-existing bugs (one
already partially fixed in a prior session, confirmed live here) and one new
cross-env bug. All fixes are code-complete and `py_compile`-clean but **not
yet committed**.

### 6.1 `generate_trajectories.py` per-env error isolation — verified live

A prior session added per-env `try/except` with `print(..., flush=True)` in
`_generate_offline()`/`main()`'s multi-env branch, so one env's failure (e.g.
its sidecar is down) doesn't abort the whole run — `merge_datasets` proceeds
with whichever envs succeeded, and `RuntimeError` is only raised if
`per_env_paths` ends up empty. **Verified live this session**: with all 4
game sidecars down, each printed
`[generate_trajectories] No data for env 'X', skipping it: Failed to init
server ... Connection refused` and was skipped; intercode alone succeeded and
the run completed with a 1-dataset merge.

Note: the single-env `generate_for_env` path still raises
`RuntimeError("No valid examples generated. Check ENVIRONMENT_SERVER_URLS.")`
if zero examples survive score-filtering for that one env — this is correct/
intended (e.g. requesting only `othello` with too few games and all losses),
not a bug.

### 6.2 `liar_dice_trajectories.py::_split_observation` — verified live

A prior session fixed `_split_observation` to search `_RE_PLAYER_LINE` /
`_RE_CURRENT_PLAYER` against the **full raw** `observation` *before* stripping
the `"Current State:\n"` prefix (the `"You are Player N."` line lives in the
preamble, before `"Current State:"`). **Verified live this session**:
`--num_games 5 --max_turn 15` against the liars_dice sidecar
(172.18.0.4:8000) → 8 valid examples (previously 0 due to this bug).

### 6.3 NEW: `tool_calls[].function.arguments` Arrow struct-schema mismatch
(blocking bug — root cause of the user's full-stack training failure)

**Symptom**: a full 5-env run (liars_dice, gin_rummy, leduc_poker, othello,
intercode) generated all 5 per-env datasets successfully, but crashed in
`merge_datasets`/`concatenate_datasets`:

```
ValueError: The features can't be aligned because the key messages of
features {..., 'tool_calls': List({..., 'function': {'name': Value('string'),
'arguments': {'command': Value('string')}}})} has unexpected type -
List(...{'arguments': {'action_id': Value('int64')}}) or Value("null").
```

This then cascaded into 5 separate training-time failures, each
`FileNotFoundError: Directory /workspace/scripts/datasets/sft_env_1 not
found` — because dataset generation never produced a usable merged dataset
for training to load.

**Root cause**: the 4 game generators build
`tool_calls[0]["function"]["arguments"] = {"action_id": <int>}`, while
`intercode_dataset.py` builds `{"command": <str>}`. `datasets`/Arrow requires
identical struct types for the same nested column path across all datasets
being `concatenate_datasets`'d — two different dict shapes for `arguments`
can't be unified. This is the exact same class of issue as the
**pre-existing** `tools` JSON-string workaround (see the comment in
`liar_dice_trajectories.py::_build_tool_example` / `intercode_dataset.py`'s
`_INTERCODE_TOOLS_JSON`), just for `tool_calls[].function.arguments` instead
of `tools[].function.parameters.properties`.

**Fix** — same JSON-string pattern, applied consistently:
- `liar_dice_trajectories.py`, `gin_rummy_trajectories.py`,
  `leduc_poker_trajectories.py`, `othello_trajectories.py`: changed
  `"arguments": {"action_id": action_id}` →
  `"arguments": json.dumps({"action_id": action_id})`.
- `intercode_dataset.py`: changed `"arguments": {"command": gold}` →
  `"arguments": json.dumps({"command": gold})`, and updated the module
  docstring's example row shape accordingly.
- `train_sft_env.py::tokenize_and_mask._process`: added a decode-back-to-dict
  step for every assistant message's `tool_calls[].function.arguments`
  (`json.loads(...)` if it's a string) *before* `apply_chat_template` — mirrors
  the existing `tools`-as-JSON-string handling. Without this, the Qwen chat
  template's `arguments | tojson` filter would double-encode an
  already-JSON-encoded string.

**Eval-alignment verification**: confirmed **token-level equality** with the
real `Qwen/Qwen2.5-3B-Instruct` tokenizer — `apply_chat_template` output
(`input_ids`) is byte-identical whether `arguments` is the old dict shape or
the new JSON-string-then-decoded shape. Rendered tool call:
`<tool_call>\n{"name": "game_action", "arguments": {"action_id": 26}}\n</tool_call><|im_end|>\n`
— i.e. the fix is purely a storage-format change to survive
`concatenate_datasets`; the trained tokens are unchanged.

**Re-validated end-to-end after the fix**: rebuilt the docker image, restarted
all 4 sidecars, reran the full 5-env `generate_trajectories.py` run → 10038-row
merged dataset, with `arguments` correctly stored as JSON strings for both
`game_action` (`'{"action_id": 4}'`) and `execute_bash`
(`'{"command": "find . -name testfile.txt"}'`). Then ran
`tokenize_and_mask` on 10 sampled rows spanning all 5 envs with the real
tokenizer — correct `<tool_call>{...}</tool_call>` rendering and correct
`assistant_masks` for both tool types.

### 6.4 Masking confirmed assistant-turn-only

Per TRL 0.27.0 source: `SFTTrainer`'s `_prepare_dataset` sees the dataset
already has `input_ids` (`is_processed = True`) and skips re-tokenization/
re-templating — it only truncates to `max_length` (packing is off). The
collator (`DataCollatorForLanguageModeling`) unconditionally applies
`output["labels"][assistant_masks == 0] = -100` when `assistant_masks` is
present, so loss is computed **only on assistant-turn tokens** (the
`game_action`/`execute_bash` tool-call JSON), exactly as `tokenize_and_mask`
constructs it. No further changes needed here.

### 6.5 Uncommitted changes (as of end of this session)

- `scripts/envs/othello_trajectories.py` — new file (§3.4).
- `scripts/envs/sft_env_configs.py` — registers `othello`,
  `_OFFLINE_ENVS = {"intercode"}`.
- `scripts/envs/generate_trajectories.py` — Othello registration + §6.1
  per-env error isolation.
- `scripts/envs/liar_dice_trajectories.py` — §6.2 `_split_observation` fix +
  §6.3 `arguments` JSON-encoding.
- `scripts/envs/gin_rummy_trajectories.py` — §6.3 `arguments` JSON-encoding.
- `scripts/envs/leduc_poker_trajectories.py` — §6.3 `arguments` JSON-encoding.
- `scripts/envs/intercode_dataset.py` — §6.3 `arguments` JSON-encoding +
  docstring update.
- `scripts/train_sft_env.py` — §6.3 `tool_calls[].function.arguments` decode
  step in `tokenize_and_mask`.

### 6.6 Suggested next steps

- Commit the above once the user is ready (repo convention: only commit when
  explicitly asked).
- The trainer container's HF Hub connectivity retries (~23s of retries before
  falling back to the local cached model) can be skipped by setting
  `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` in the trainer's environment,
  since `/cache/models/Qwen--Qwen2.5-3B-Instruct` is pre-downloaded and no
  internet is available on `internal_bridge` anyway.
- Run a real training job (GPU) on the 5-env merged dataset to unblock §4
  item 2 (`pvp_smoke_match.py` checkpoint comparison).
