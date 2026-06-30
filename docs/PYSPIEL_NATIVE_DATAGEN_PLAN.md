# Native pyspiel data generation — implementation plan

Status: **proposal, not yet implemented**. Written for review before any code changes.

**Non-negotiable constraint**: this change touches *only* how game state
reaches each generator (HTTP+regex → direct `pyspiel.State`). It must not
change which move gets picked. See §3.3a for exactly what counts as "policy"
(unchanged) vs. "plumbing" (replaced).

## 1. The problem with the current pipeline

This repo never spins up the env-server sidecars itself — the surrounding
tournament/training procedure (the trainer/orchestrator that calls into this
image, mirrored locally by `~/sn56-G.O.D-env/examples/run_environment_task.sh`
+ `run_env_sidecars.sh`) starts a small, fixed pool of `gradientsio/mcts-api`
**HTTP sidecar containers** per training job and hands this repo their URLs
via `ENVIRONMENT_SERVER_URLS`. This repo only ever *consumes* that pool
(`scripts/envs/shared_env.py::init_env_pool`) — it has no control over how
many sidecars exist (production: one per GPU, so typically small, e.g. 4).
Today, `scripts/envs/generate_trajectories.py` generates SFT rows by:

1. Connecting to that fixed, externally-provided pool of sidecars.
2. Each worker process round-robins across that fixed pool, `POST /reset` +
   `POST /step`-ing a raw-text observation back and forth.
3. Per-game generators (`liar_dice_trajectories.py`, `gin_rummy_trajectories.py`,
   `leduc_poker_trajectories.py`, `othello_trajectories.py`) regex-parse that
   raw text (`_format_observation` / `split_normalized_observation`) back into
   `(state_desc, player_id, legal_actions)` so `pvp_format.py` can rebuild the
   eval-shaped `{system, user, assistant}` row.
4. `goofspiel` and `liars_dice` have already abandoned the server entirely
   (`goof_spiel_trajectories.py`, `liar_dice_trajectories.py`) because the
   server was found to be **wrong** for goofspiel (corrupts simultaneous-move
   state from round 1) and unhelpfully weak/server-bottlenecked for liars_dice,
   replacing it with from-scratch hand-rolled game simulators in pure Python
   (`goof_spiel_minimax.py`, `liar_dice_policy.py`) that reimplement dice/card
   rules independently of `pyspiel`. These two were the right call given what
   was available at the time, and their *policies* are confirmed-good and
   **stay exactly as they are** — but their underlying dice/card mechanics are
   currently hand-rolled Python, where `pyspiel`'s actual `liars_dice`/
   `goofspiel` implementations are C++ underneath. Rebasing their plumbing
   onto real `pyspiel.State` objects (§3.3a) gets both games the same
   correctness guarantee the other three get from this plan, plus a speed
   win on top (faster legal-action/apply-action calls, see §3.5/§6 timing).

This has three concrete costs:

- **Throughput ceiling**: only as many concurrent HTTP connections as the
  externally-provided sidecar pool has (production: one per GPU, so a small,
  fixed number) regardless of how many CPU cores the training container
  itself has available — confirmed in `generate_trajectories.py`'s own
  comment ("servers are a fixed-throughput shared resource ... 2x workers ~=
  2x speed"). `_ENV_DEFAULT_ARGS` asks for 4–200k games per env; this is the
  actual wall-clock bottleneck.
- **A text-roundtrip layer that's pure overhead**: raw text → regex parse →
  rebuilt structured row, when the eval harness (`~/sn56-G.O.D-env/core/pvp/bot.py`)
  builds that row directly from a live `pyspiel.State`, never going through
  text in the middle except as the *rendered prompt itself*.
- **Correctness risk from divergence**: the server is a black box
  (`gradientsio/mcts-api`) we don't control, already shown buggy for goofspiel
  and reimplemented from scratch for two of five games. Every hand-rolled
  reimplementation (`othello_bitboard.py`, `othello_minimax.py`,
  `gin_rummy_refined.py`, `gin_rummy_opponent_modeling.py`,
  `leduc_poker_opponent_modeling.py`, `goof_spiel_minimax.py` — ~4,500 lines
  combined) is a second, independent implementation of rules that `pyspiel`
  already implements canonically, and that the real eval harness actually
  runs against.

## 2. What the eval harness actually does (ground truth)

Confirmed by reading `~/sn56-G.O.D-env/core/pvp/{bot,agents,game_eval,baseline,scoring}.py`:

- **No HTTP server at all.** Eval, the PvP matchup runner
  (`validator/evaluation/pvp/game_runner.py`), and even the **MCTS baseline**
  (`core/pvp/baseline.py`) all load games via `pyspiel.load_game(...)` and
  drive them **in-process** with OpenSpiel's own `evaluate_bots` /
  `pyspiel.Bot` interfaces. The `mcts-api` Docker image is apparently a
  separate, older component that training currently depends on but eval does
  not.
- **Per-game agent classes** (`core/pvp/agents.py`, one `BaseGameAgent`
  subclass per game: `LiarsDiceAgent`, `LeducPokerAgent`, `GinRummyAgent`,
  `OthelloAgent`, `GoofspielAgent`) own exactly three things:
  - `generate_params(config_id) -> GameParams` — picks the pyspiel game
    variant (e.g. `GinRummyParams(hand_size=7+var, knock_card=10-var)`,
    `GoofspielParams(num_cards=...)`).
  - `load_game(params) -> pyspiel.Game` — usually
    `pyspiel.load_game(game_name, params.to_pyspiel())`; Goofspiel overrides
    it to wrap with `pyspiel.convert_to_turn_based(...)` since OpenSpiel's
    native goofspiel is simultaneous-move.
  - `format_state(state, player_id) -> str` — the **state-description** text
    in the user prompt. Mostly `state.observation_string(player_id)` or
    `state.information_state_string(player_id)`, with light per-game
    post-processing (Liar's Dice extracts dice/bid into readable lines;
    Othello prepends `"You play x (Black)."`; Goofspiel prepends
    `"You are Player N."`).
  - `setup_initial_state(state, seed)` — only Othello overrides this, to
    apply 2–6 seeded random opening plies (it has no chance nodes, so every
    game would otherwise start identical).
- **`LLMBot._run_turn`** (`core/pvp/bot.py`) builds the *exact* 2-message
  turn that training is trying to reproduce:
  ```python
  legal_actions = state.legal_actions(self._player_id)
  messages = [system_prompt, user_prompt(state, legal_actions)]
  tools = memory_tools + [game_action_tool(legal_hint, legal_actions)]
  ```
  where `user_prompt` is:
  ```python
  f"Current state:\n{agent.format_state(state, player_id)}\n\n"
  f"You are Player {player_id}.\n"
  f"Legal actions:\n" + "\n".join(f"{a} -> {state.action_to_string(player_id, a)}" for a in legal_actions)
  ```
  — i.e. **action labels come from `state.action_to_string`, not from
  server-rendered text**. `system_prompt` = `agent.generate_system_prompt()`
  (rules) + memory block + `TOOL_GUIDANCE`, byte-for-byte what
  `scripts/envs/pvp_format.py::build_full_system_prompt` already reproduces.
- **The MCTS opponent is also in-process** (`core/pvp/baseline.py::_make_mcts_bot`):
  ```python
  evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=...)
  mcts.MCTSBot(game, uct_c=2.0, max_simulations=N, evaluator=evaluator, random_state=...)
  ```
  using `open_spiel.python.algorithms.mcts` — the *same* `pyspiel.Game` object
  the LLM-side bot plays against, no server involved, and not capped at 4
  concurrent instances — it's just a Python object, you can have as many as
  you have CPU.
  - Note: this in-process MCTSBot's exact engineering — and the resulting
    play-strength — may differ from the (also `MCTSBot`-named) opponent the
    `mcts-api` server provides, since the server's implementation is opaque to
    us. This needs empirical validation (§6), not assumed equivalence.
- **Outcome scoring** (`core/pvp/scoring.py::determine_outcome`) is a pure
  function of `state.returns()`, zero-sum-normalized to win/loss/draw —
  exactly the `(0.0, 0.5, 1.0)` reward scale `generate_trajectories.py`
  already expects from generators for `--sample-by-score`.
- **Game/config sampling** (`core/pvp/game_eval.py::config_id_for_seed`) is a
  pure, deterministic `seed -> config_id` function (`Random(seed).randint(...) % PVP_CONFIG_ID_DIVISOR`)
  — easy to mirror so SFT data and eval draw from the same variant
  distribution.

This means: **everything the current generators do by parsing server text can
instead be read directly off a live `pyspiel.State`**, with zero text
round-trip, and the opponent can be `pyspiel`'s own `MCTSBot` running locally
with no server and no concurrency cap.

## 2.1 Important risk: per-game config may not match what the sidecar used

This needs calling out explicitly because it's easy to get wrong silently.

Today, `othello`/`leduc_poker`'s generators send only `{"task_id": game_id}`
to the sidecar's `/reset` and let the **server** decide the game variant
(board size, hand size, knock card, etc.) from that id — we don't have the
`mcts-api` server's source, so we don't actually know its `task_id -> variant`
mapping. `gin_rummy`'s generator goes further and reverse-engineers the
variant *after the fact* by regexing it back out of the server's rendered
text (`parse_knock_card`). In other words: **the current pipeline's per-game
variant distribution is defined by an opaque, unverified server**, and we've
been trusting it matches eval's distribution without being able to check.

Eval's actual mapping is not opaque — it's `core/pvp/game_eval.py::config_id_for_seed`
(`Random(seed).randint(task_id_min, task_id_max) % PVP_CONFIG_ID_DIVISOR`)
feeding `agent.generate_params(config_id)` (e.g. `GinRummyAgent`: `hand_var =
(config_id // 3) % 3`, `knock_var = config_id % 3` → `hand_size = 7..9`,
`knock_card = 8..10`; `GoofspielAgent`: `num_cards` cycles through `(5, 8, 10,
13)`). This plan's recommendation is to **copy `core/pvp/agents.py` and
`core/pvp/game_eval.py`'s `config_id_for_seed` directly** (per the go-ahead to
copy G.O.D code when it's faster and safer than reimplementing) rather than
trying to reverse-engineer or guess at the server's mapping further. This is
strictly an improvement on the status quo — it's ground truth instead of a
guess — but it does mean the per-game variant distribution sampled for SFT
data may shift slightly from whatever the server was actually doing, which is
worth a quick before/after sanity check (§6) rather than assuming it's a
no-op.

## 3. Proposed architecture

### 3.1 New shared module: `scripts/envs/pvp_game_engine.py`

Port (not re-derive) the following from `~/sn56-G.O.D-env/core/pvp/`, trimmed
to what data generation needs (no `LLMBot`, no chat/tool-call parsing, no
SGLang):

- `BaseGameAgent` + the 5 concrete agents (`LiarsDiceAgent`, `LeducPokerAgent`,
  `GinRummyAgent`, `OthelloAgent`, `GoofspielAgent`) — verbatim port of
  `core/pvp/agents.py`. This file already has **zero** dependency on the chat/
  tool-calling machinery; it only needs `pyspiel` + `core/models/pvp_models.py`'s
  `GameParams` subclasses (also a trivial, dependency-free port — pure
  pydantic models with a `to_pyspiel()` dict-dump).
- `config_id_for_seed` — verbatim port of `core/pvp/game_eval.py`'s one
  function (depends only on `core/constants.py`'s per-env `task_id_min/max`,
  which `scripts/envs/shared_env.py::GAMES_TO_TASK_ID_RANGE` already mirrors).
- `determine_outcome` — verbatim port of `core/pvp/scoring.py` (one pure
  function, already need-for-need identical to the `final_reward` semantics
  `generate_trajectories.py` expects).
- A **local MCTS opponent factory** mirroring `core/pvp/baseline.py::_make_mcts_bot`:
  ```python
  from open_spiel.python.algorithms import mcts
  def make_mcts_bot(game, simulations, seed):
      evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=np.random.RandomState(seed))
      return mcts.MCTSBot(game, uct_c=2.0, max_simulations=simulations, evaluator=evaluator, random_state=np.random.RandomState(seed))
  ```

This module becomes the **single source of truth for game mechanics**,
replacing the bespoke per-env reimplementations. It depends on `pyspiel` (pip
package `open-spiel`, **already used and pinned to `==1.6.13`** by
`~/sn56-G.O.D-env/dockerfiles/model-prep.dockerfile` and `pvp-eval.dockerfile`
— see §5 for the exact version/compatibility this repo should pin) and nothing
else project-specific beyond what's already in `scripts/envs/`.

### 3.2 New per-turn row builder: replaces `_format_observation` / `split_normalized_observation`

Today: `raw_text → regex → (state_desc, player_id, legal_actions) → build_user_prompt`.

New: directly off the `pyspiel.State`:

```python
def build_turn_example(agent, state, player_id, memories=None):
    legal_actions = state.legal_actions(player_id)
    state_desc = agent.format_state(state, player_id)
    legal_pairs = [(a, state.action_to_string(player_id, a)) for a in legal_actions]
    system = build_full_system_prompt(agent.game_name, memories)   # unchanged, pvp_format.py
    user = build_user_prompt(state_desc, player_id, legal_pairs)   # unchanged, pvp_format.py
    tools = build_pvp_tools([a for a, _ in legal_pairs])           # unchanged, pvp_format.py
    return system, user, tools, legal_actions
```

**`pvp_format.py`, `pvp_tools.py`, `pvp_memory.py`, `pvp_models.py`,
`pvp_constants.py`, `pvp_assets/pvp_game_prompts.yml` are all unaffected** —
they already operate on `(state_desc, player_id, legal_actions)` tuples, which
is exactly what `pyspiel.State` gives for free, with no parsing. This deletes
`_format_observation`/`split_normalized_observation`'s reason for existing
(the regex layer), not `pvp_format.py`'s prompt-building, which stays as the
single source of truth for prompt text either way.

### 3.3 Per-game generator rewrite — same policy logic, new state access

Each of the 5 generators currently has two halves: (a) **how to read/parse the
opponent server's state**, and (b) **what move to pick given that state** (the
"expert"/heuristic/MCTS-probability policy). **Only (a) changes — (b) is the
trained label and must produce the identical decision given identical game
state, full stop.** Every one of these currently computes its features by
regex-scraping rendered text (e.g. `gin_rummy_trajectories.py::parse_discard_pile`,
`othello_trajectories.py`'s 2-char-label regex, `liar_dice_policy.py`'s
dice/bid string parsing) — once we hold the actual `pyspiel.State`, those
features are direct method calls (`state.legal_actions()`,
`state.information_state_tensor()`, hand/discard pile accessors, etc.),
which is strictly more reliable than text-scraping the same information back
out of a rendering, but the *decision function itself* (the actual `choose_*`/
heuristic-scoring code) is ported with its logic untouched — only its inputs
change from "parsed strings" to "direct state accessors".

| Game | Policy (kept as-is) | Plumbing being replaced | Notes |
|---|---|---|---|
| `liars_dice` | `liar_dice_policy.choose_action` — MCTS-probability softmax. **Unchanged.** | `_split_observation`'s regex parse of server text → direct `pyspiel.load_game("liars_dice", ...)` state (dice, bid history via `state.information_state_string`/`state.history()`) | Removes the hand-rolled dice/bid reimplementation in favor of real `pyspiel` state; opponent becomes the local `MCTSBot` (§3.1) instead of `liar_dice_policy`'s own opponent stand-in — verify comparable strength, §6 |
| `gin_rummy` | hand-coded deadwood/meld heuristic in `gin_rummy_trajectories.py`. **Unchanged.** | `extract_and_format_observation`'s regex parse → direct hand/discard/meld state off `pyspiel`'s real `gin_rummy` game | `pyspiel`'s gin_rummy already exposes deadwood/meld primitives some of `gin_rummy_refined.py`/`gin_rummy_opponent_modeling.py` reimplement — audit for direct replacement vs. porting the heuristic's call sites onto equivalent `pyspiel` accessors |
| `leduc_poker` | uniform-random policy over legal actions. **Unchanged.** | server text parse → `pyspiel.load_game("leduc_poker", ...)`, `state.legal_actions()` | Simplest port — already random, no heuristic to preserve |
| `othello` | corner/edge heuristic in `othello_trajectories.py`. **Unchanged.** | `othello_bitboard.py`/`othello_board.py` reimplemented board → `state.legal_actions()`/`state.action_to_string()` (already 2-char cell labels, confirmed via direct test — see §6) off `pyspiel`'s real `othello` game | Likely **deletes** `othello_bitboard.py`/`othello_minimax.py`/`othello_minimax_selfcheck.py` (518+171+139 lines) entirely if the heuristic's board-feature needs (corner/X-square/edge classification) can be expressed against `pyspiel`'s state directly |
| `goofspiel` | minimax/equilibrium solve in `goof_spiel_minimax.py`. **Unchanged.** | hand-rolled local card simulator → `pyspiel.convert_to_turn_based(pyspiel.load_game("goofspiel", ...))` (confirmed working, §6), mirroring `GoofspielAgent.load_game` exactly | Closes the exact bug this generator was built to route around — now using the *real* engine (C++-backed) instead of a hand-rolled Python stand-in for it |

In every case the **expert/heuristic decision logic itself is reusable
verbatim** — what changes is only the plumbing that feeds it game state. The
concrete contract for the port: take each `choose_*`/`get_*_action` function,
keep its body's *scoring/selection* logic identical, and only change how its
input features (hand, discard pile, dice, board, deadwood, etc.) are computed
— from regex-over-rendered-text to direct `pyspiel.State` method calls. Any
PR implementing this should be reviewable as "plumbing diff, zero policy
diff," and §6 item 4 below exists specifically to catch any accidental
decision drift.

### 3.4 Generation loop — replaces the HTTP `/reset`+`/step` cycle

```python
def generate_episode(agent, game, opponent_bot, expert_policy, seed, max_turn):
    state = game.new_initial_state()
    agent.setup_initial_state(state, seed)
    examples = []
    for _ in range(max_turn):
        if state.is_terminal():
            break
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(np.random.choice(outcomes, p=probs))
            continue
        cur = state.current_player()
        if cur == expert_seat:
            system, user, tools, legal = build_turn_example(agent, state, cur)
            action = expert_policy(state, cur, legal)          # existing heuristic, re-pointed at state
            examples.append(_build_tool_example(system, user, tools, action))
        else:
            action = opponent_bot.step(state)                  # pyspiel.Bot (MCTSBot) — in-process
        state.apply_action(action)
    final_reward = score_for(state, expert_seat)                # core/pvp/scoring.py port
    return examples, final_reward
```

This is a **pure-Python, in-process loop** — no HTTP, no JSON envelope, no
container. `generate_trajectories.py`'s outer structure (process pool,
score-based sampling, per-env defaults, multi-env merge) needs almost no
changes — only `_worker_init`/`_worker_play` stop taking an `endpoint: str`
(there's no server URL anymore) and `_NO_SERVER_ENVS`/`init_env_pool`'s whole
reason for existing goes away for **all 5** games, not just 2.

### 3.5 Concurrency — exactly how the worker count is computed

Today's worker count is `num_workers or max(1, num_servers)` — bounded by
however many sidecars the orchestrator happened to start (production: ~4,
one per GPU), regardless of how many CPU cores the training container itself
has. Once there's no server in the loop, every env becomes shaped like
today's already-`_NO_SERVER_ENVS`-exempted `goofspiel`/`liars_dice`
(`generate_trajectories.py`'s existing `num_workers or max(1, os.cpu_count()
or 4)`), and that formula extends to all 5 games, not just 2.

**Concretely:**

- **Single-env run**: `num_workers = os.cpu_count()`, one OS process per core
  via the existing `ProcessPoolExecutor`, capped at `num_games` so a small
  batch doesn't spin up idle workers (`min(os.cpu_count(), num_games)`).
  Each worker process plays games sequentially — no need for additional
  threading inside a worker, since each `pyspiel` call (game step, MCTS
  rollout) is independent CPU work with no I/O wait to hide.
- **Multi-env concurrent run** (today's `main()` multi-env branch divides the
  *server pool* across envs): replace that division with dividing **CPU
  cores** across the envs running concurrently —
  `per_env_workers = max(1, os.cpu_count() // len(game_env_names))`, the same
  shape as today's `shared = num_servers // len(server_bound_envs)` line in
  `generate_trajectories.py::main()`, just driven by core count instead of
  server count. `--num_workers` still overrides this explicitly when given.
- Leave 1 core free for the orchestrating main process / OS scheduling
  overhead is optional polish, not required — `ProcessPoolExecutor` workers
  are I/O-light and short-lived per game, so slight oversubscription has
  negligible cost; default to the simple `os.cpu_count()` form unless
  profiling says otherwise.

**Measured, not estimated** (§6.5): the actual production trainer image
(`standalone-text-trainer:latest`, already built and cached locally) reports
`os.cpu_count() == 30` inside its `.grpo_env` venv. Today's per-env worker
count is capped at the sidecar pool size (production: ~4 per GPU). That's
roughly a **7x+ increase in concurrent games-in-flight** from the worker-count
change alone, before counting the per-call speedup from C++ `pyspiel` state
operations replacing Python regex parsing (§6.5 also has a directly-measured
MCTS timing: 25-simulation `MCTSBot.step()` averaged ~22ms per call inside
that same container).

## 4. What gets deleted / heavily shrunk

- `init_env_pool`'s HTTP warm-up path, `_NO_SERVER_ENVS` special-casing
  (`shared_env.py`, `generate_trajectories.py`) — no servers left to special-case
  around.
- `_format_observation`/`split_normalized_observation`'s regex parsing in
  `othello_trajectories.py`, `leduc_poker_trajectories.py`,
  `gin_rummy_env.py::extract_and_format_observation`, `liar_dice_trajectories.py::_split_observation`.
- Candidate full deletions, pending the audit in §3.3 row-by-row: `othello_bitboard.py`,
  `othello_minimax.py`, `othello_minimax_selfcheck.py` (othello board logic
  pyspiel already provides); large parts of `gin_rummy_refined.py` /
  `gin_rummy_opponent_modeling.py` / `leduc_poker_opponent_modeling.py` if
  their state-tracking duplicates what `pyspiel`'s state already exposes
  (these need a closer read than this pass did — flag, don't delete blind).
- `examples/run_env_sidecars.sh`-equivalent usage from this repo's docs/scripts
  (training repo doesn't own that file, but any reference to spinning up
  sidecars for SFT datagen goes away).
- `ENVIRONMENT_SERVER_URLS` dependency for SFT datagen specifically (GRPO env
  rollout, `train_grpo_env.py`/`env_configs.py`, is **out of scope** — it's a
  separate, not-yet-tool-calling-aligned system per
  `docs/SFT_ALIGNMENT_PLAN.md` §5.2/§5.3 and may still want live servers for
  its own reasons; don't couple this change to it).

## 5. What's required but new

- **Dependency, confirmed**: `open-spiel==1.6.13` (the version
  `~/sn56-G.O.D-env/dockerfiles/model-prep.dockerfile`/`pvp-eval.dockerfile`
  pin) requires **Python >= 3.11** — PyPI's release metadata shows the
  `>=3.11` floor started at `1.6.12`; `1.6.6`–`1.6.11` support `>=3.10`, and
  nothing on PyPI supports 3.9 or below for the `1.6.x` line. This matters
  because **the actual training path is not what `scripts/training_requirements.txt`
  suggests**: `generate_trajectories.py`/`train_sft_env.py` both run inside
  `dockerfiles/standalone-text-trainer.dockerfile`'s `/workspace/.grpo_env`
  venv (built from `scripts/grpo_requirements.txt`, activated by
  `run_text_trainer.sh` before `text_trainer.py` ever calls
  `tokenize_cmd`/`generate_cmd`/`train_cmd` as subprocesses) —
  `training_requirements.txt` isn't installed by that Dockerfile at all
  (it's used by a different, InstructTextTask-only venv path, `/workspace/axo_py`).
  So the dependency needs to go into **`scripts/grpo_requirements.txt`** and
  `standalone-text-trainer.dockerfile`'s existing install step, not
  `training_requirements.txt`.
- **Verified directly against the real, already-built image** (see §6.5 for
  the exact commands/output): `standalone-text-trainer:latest`'s
  `.grpo_env` venv runs **Python 3.12.3** — comfortably above open-spiel's
  `>=3.11` floor — and `pip install open-spiel==1.6.13` inside that exact
  container installs cleanly (`manylinux_2_27/2_28` prebuilt wheel, no
  compilation, ~15.5MB), pulling in only one new transitive dependency
  (`ml-collections`), with **no version conflicts** against the existing
  torch/transformers/trl/peft stack. `import pyspiel` and a live
  `MCTSBot` rollout both work post-install in that environment.
- **Ported files** (new, under `scripts/envs/`): `pvp_game_engine.py`
  (agents + `config_id_for_seed` + `determine_outcome` + `make_mcts_bot`, per
  §3.1), and trimmed local copies of `core/models/pvp_models.py`'s
  `GameParams` subclasses (or fold into the existing
  `scripts/envs/pvp_models.py`, which already holds the *other* pvp_models —
  check for naming collisions before merging).
- **`core/config/pvp_game_prompts.yml` parity check**: confirm
  `scripts/envs/pvp_assets/pvp_game_prompts.yml` is still byte-identical to
  the source (it was a verbatim copy at groundwork time, §1 of
  `SFT_ALIGNMENT_PLAN.md`) — game rules text must stay in sync since this
  plan doesn't touch how it's loaded, only how state reaches it.

## 6. Validation plan (must happen before trusting the new generators)

This is a rewrite of the data-generation substrate for every game env, so
correctness validation is not optional:

1. **Engine parity smoke test**: for each game, load via this repo's new
   `pvp_game_engine.py` agent and separately via
   `~/sn56-G.O.D-env/core/pvp/agents.py`'s real agent with the **same seed**;
   assert `generate_params`/`load_game`/initial `format_state` produce
   identical output. Cheap, catches accidental drift in the port immediately.
2. **Prompt byte-equality**: re-run the existing spot-check this repo already
   did once per `SFT_ALIGNMENT_PLAN.md` §4 — compare a handful of generated
   rows against `~/sn56-G.O.D-env/examples/pvp_eval_outputs`/
   `pvp_smoke_all.json`-style real eval trajectories, but now there's no
   server in the loop to introduce drift, so this becomes a pure
   `pvp_format.py` regression check, not an integration test.
3. **Opponent strength sanity**: the in-process `MCTSBot` (§3.1) is not
   guaranteed to play identically to whatever `mcts-api`'s server-side "mcts"
   opponent actually was (closed-box). Run a batch of games with both at
   matched `max_simulations` and compare win rates / game-length
   distributions before fully cutting over — a meaningfully weaker or
   stronger in-process opponent changes the difficulty of the SFT label
   distribution.
4. **Per-game heuristic regression**: for `gin_rummy`/`othello` (the two with
   nontrivial hand-coded policies), run the ported heuristic against a fixed
   seed set and diff its chosen actions against the current generator's
   choices on the *same* underlying game trajectory — should be identical
   modulo any deliberate bugfix, since the decision logic isn't supposed to
   change, only its plumbing.
5. **Throughput measurement**: confirm the actual wall-clock win — benchmark
   games/sec before (4-server HTTP pool) vs. after (CPU-core-bound
   `ProcessPoolExecutor`) on the same machine, for at least one
   network-bound game (`othello`) and confirm it scales near-linearly with
   `os.cpu_count()`.
6. **Full pipeline run**: regenerate all 5 envs + merge
   (`generate_trajectories.py`'s multi-env path), run `tokenize_and_mask`
   on a sample, confirm `assistant_masks`/`tool_calls` shape is unchanged
   from today's output (this part of the pipeline is untouched by this plan
   and should produce byte-identical *row shape*, just sourced differently).

### 6.5 Already done during this planning pass (evidence, not just claims)

Run directly against `standalone-text-trainer:latest` (the real, already-built
production image cached on this machine) and a throwaway `uv`-managed venv,
to de-risk the dependency question before committing to this plan:

- `docker run --rm --entrypoint /workspace/.grpo_env/bin/python3 standalone-text-trainer:latest --version`
  → `Python 3.12.3`.
- `docker run --rm --entrypoint /workspace/.grpo_env/bin/pip standalone-text-trainer:latest install --no-cache-dir open-spiel==1.6.13`
  → installs cleanly, prebuilt wheel
  (`open_spiel-1.6.13-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`),
  one new transitive dep (`ml-collections`), no conflicts reported against
  the existing environment.
- Inside that same container, post-install: `import pyspiel` succeeds;
  `pyspiel.load_game("othello")`, `"leduc_poker"`, `"gin_rummy"` (with
  `{"hand_size": 7, "knock_card": 10}`), and
  `pyspiel.convert_to_turn_based(pyspiel.load_game("goofspiel", {...}))` all
  load without error; `state.action_to_string(0, 19)` on a fresh `othello`
  state returns `"d3"`, matching the 2-char cell-label format the current
  `othello_trajectories.py` heuristic already regexes for — confirming the
  port target's label format is unchanged.
- `os.cpu_count()` inside the container: **30**.
- Timing: 20 sequential `MCTSBot.step()` calls (`uct_c=2.0`,
  `max_simulations=25`, `RandomRolloutEvaluator(n_rollouts=1)`) on a fresh
  `othello` state took 0.454s total inside the container (~22.7ms/call) —
  cheap enough that per-move opponent cost is not expected to be the new
  bottleneck even at much higher worker counts.
- (Separately, in a local `uv` venv pinned to Python 3.10: confirms
  `open-spiel==1.6.13` is *not* installable there — `uv` correctly resolves
  and rejects it on the `>=3.11` floor, falling back to `1.6.11` if
  unpinned. This is exactly why the version note in §5 matters — pin
  `==1.6.13` deliberately, don't leave it floating, since an unpinned
  install on a future, different base image could silently downgrade.)

Not yet done (still needs real implementation + the rest of §6.1–§6.4):
engine-parity smoke test against the real `core/pvp/agents.py`, prompt
byte-equality against recorded eval trajectories, opponent-strength
comparison against the old `mcts-api` server's "mcts" option, and the
per-game heuristic decision-regression diff.

## 7. Open questions to resolve before implementation

- Does `pyspiel`'s `gin_rummy`/`othello` implementation expose every feature
  the existing heuristics need (deadwood computation, meld detection, corner/
  X-square classification) directly, or do those heuristics need to keep some
  bespoke feature-extraction code, just now reading from `state` instead of
  text? (§3.3 flags this per-game; needs a focused read of `pyspiel`'s game
  source / Python bindings for `gin_rummy` and `othello` specifically.)
- Is `liar_dice_policy.py`'s current "local sim" *itself* already a faithful
  reimplementation of `pyspiel`'s `liars_dice`, or did it diverge in ways that
  happen to not matter for training? Worth a diff before assuming the port is
  risk-free.
- Should the in-process opponent always be `MCTSBot` (matching what
  `core/pvp/baseline.py` uses for the official baseline), or does any
  generator benefit from a cheaper/different opponent now that we're not
  paying a server round-trip per move? (e.g. raising `mcts_simulations` for
  free since there's no longer a fixed server pool to overload.) Per the
  constraint in §3.3a, this only applies to the *opponent's* moves — the
  expert/teacher side's policy is fixed and out of scope for this question.
- Confirm whether `train_grpo_env.py`/GRPO rollouts share any of the touched
  files (`shared_env.py`, `env_configs.py`) such that deleting
  `_NO_SERVER_ENVS`/`init_env_pool` plumbing needs to keep a GRPO-only path
  alive — scope this change to SFT datagen only if so.
- §2.1's variant-distribution shift (server's opaque `task_id` mapping vs.
  eval's real `config_id_for_seed`/`generate_params`): confirm this is
  acceptable to land as-is (it's a correctness improvement, not a bug) rather
  than something that needs to be reconciled with already-trained checkpoints'
  data distribution.
