# Clobber SFT Data Generation — Strategy & V1 Plan

Clobber is a new PvP env added on the eval side (`~/sn56-G.O.D-env`, branch
`game/clobber`, see `docs/clobber_implementation.md` there). This doc covers
how to generate SFT training data for it, following the same pyspiel-native
pattern established in `docs/PYSPIEL_NATIVE_DATAGEN_PLAN.md` for the other 5
envs (liars_dice, gin_rummy, leduc_poker, othello, goofspiel).

Backed by OpenSpiel's built-in `clobber` game: two players move a piece onto
an orthogonally-adjacent enemy piece, capturing it; whoever can't move loses.
No chance nodes, strict turn alternation — mechanically closest to Othello
among the existing envs. Board sizes cycle through `(4,5)`, `(5,5)`, `(5,6)`
via `config_id % 3` (`ClobberAgent.generate_params`, ported from
`core/pvp/agents.py`).

## Why material counting doesn't work here

Clobber belongs to the "all-small games" class in combinatorial game theory:
the win condition is "last player able to move wins," not piece count. A
pile of pieces with no adjacent enemy is dead weight — it can't capture and
can't be captured, contributing nothing to either side's mobility. Any
evaluation heuristic has to be about move availability, not material.

## Research: what's promising, what isn't

A draft strategy doc proposed several named "principles" for evaluation.
Checked each against the actual literature:

- **"Lorvi Concentration Strategy" (minimize own group count) — verified
  real.** Imre Lorvi was an actual competitive Clobber AI author; this was
  his documented heuristic. The general "maximize same-color neighbor-pair
  density" idea is also independently attested in the literature as a
  strategy used by successful Clobber programs.
- **"Kosik-Sutt Cluster Principle" — could not verify, likely fabricated.**
  No trace of this name in the actual CGT/Clobber research lineage (Albert,
  Grossman, Nowakowski invented the game in 2001; the research since is
  well-documented and this isn't in it). The underlying mechanic it
  describes happens to overlap with the real "neighbor-pair density" idea
  above, but the citation itself isn't trustworthy — a reminder that named
  "principles" in research-summary docs like this need checking, not
  assuming, since fabricated-but-plausible citations are a known failure
  mode.
- **"Dead-Zone Avoidance" — real mechanic, but likely redundant with plain
  mobility difference.** A piece with no adjacent enemy already contributes
  zero to a player's `legal_actions()` count, so a current-ply
  mobility-difference heuristic already captures most of this implicitly.
  Treating it as a separate weighted term would mostly matter as a
  *potential future* mobility / lookahead concept (akin to Othello's
  frontier-vs-actual-mobility distinction) — worth testing as a v2
  refinement, not needed for v1.
- **"Symmetry Copycat" — drop entirely, for a stronger reason than board
  size.** None of the 3 board sizes (4×5, 5×5, 5×6) are the right symmetric
  shape anyway, but more fundamentally: mirroring strategies need the move
  set to preserve the pairing move-for-move, and Clobber's capture rule
  removes pieces asymmetrically over time (capturing your mirror-pair
  counterpart doesn't simultaneously remove the piece that captured it).
  This breaks the simple "Tweedledum-Tweedledee" pairing argument even on a
  symmetric starting board.
- **Subgame decomposition (CGT atomic weights) — real, well-published,
  genuinely the strongest idea in the doc.** Confirmed via Claessen's
  Maastricht thesis and a follow-up paper ("Combining Combinatorial Game
  Theory with an α-β Solver for Clobber"): once the board fragments into
  disconnected components, you can build an endgame database of each
  component's exact CGT value ("atomic weight") and feed it into alpha-beta
  instead of re-searching the whole fragmented position from scratch. The
  practical version we'd actually want is simpler than full atomic-weight
  math: BFS/DFS to detect disconnected components on the occupied-cell
  adjacency graph, then alpha-beta only within whichever component contains
  the move under consideration. This is a real depth multiplier for the
  endgame — but per the throughput numbers below, not needed for v1
  correctness, since plain whole-board alpha-beta is already affordable for
  the full game length on these board sizes. Worth adding later if depth
  turns out to be the bottleneck.

Sources checked:
- [COMBINATORIAL GAME THEORY IN CLOBBER — Jeroen Claessen Master Thesis, Maastricht University](https://project.dke.maastrichtuniversity.nl/games/files/msc/Claessen_thesis.pdf)
- [Combining Combinatorial Game Theory with an α-β Solver for Clobber: Theory and Experiments (Springer)](https://link.springer.com/chapter/10.1007/978-3-319-67468-1_6)
- [Combining Combinatorial Game Theory with an α-β Solver for Clobber (ResearchGate)](https://www.researchgate.net/publication/309668700_Combining_Combinatorial_Game_Theory_with_an_a-b_Solver_for_Clobber)
- [Using Combinatorial Solutions and Atomic Weights to Play Competitive AI Clobber (IEEE)](https://ieeexplore.ieee.org/document/8959940/)
- [An introduction to Clobber (ResearchGate)](https://www.researchgate.net/publication/227859257_An_introduction_to_Clobber)

## MCTS vs minimax for the teacher role

OpenSpiel exposes two different MCTS implementations: the
`open_spiel.python.algorithms.mcts.MCTSBot` this repo already uses everywhere
(only the game engine calls are C++; the search loop itself is pure Python),
and a separate real C++-bound `pyspiel.MCTSBot` + `pyspiel.RandomRolloutEvaluator`
(confirmed present in the venv) that would be genuinely faster if used. So
"MCTS can be faster" is true — but raw speed isn't the deciding factor for
the *teacher* role specifically.

Clobber's win condition ("who runs out of moves first") doesn't correlate
cleanly with uniform-random rollouts the way Othello's disc-count does — a
position that's actually bad from a controlled-mobility standpoint can look
fine under random-rollout noise unless given a lot of simulations. A
mobility-difference heuristic encodes the thing that actually decides the
game directly, and Clobber's boards are tiny enough that alpha-beta can
search deep (see benchmark below) — which plays to minimax's strength
(exact within depth) more than MCTS's (built for huge branching factors,
which Clobber doesn't have).

**Decision: keep the same split every other game in this repo already
uses** — minimax-with-mobility-heuristic as the **teacher** (controls
training-data quality directly), MCTS as the **opponent** (sparring
partner, and matches the real eval-time baseline — `eval_payload_extra:
{"opponent": "mcts", ...}` in the eval repo's `ENVIRONMENT_CONFIGS`, so
training-time opponent behavior matches what the model will actually face).

## Measured search depth/cost (why this is computationally fine)

Two structural facts make Clobber much cheaper to search than Othello:

1. **Branching is small and shrinks fast.** Every move captures exactly one
   piece, so total game length is hard-capped at `cells - 1` plies (19 / 24
   / 29 for the 3 board sizes). Branching factor *peaks* at the opening
   (= total grid edges: 31 / 40 / 49) and *averages* much lower over a real
   game (measured via random playout: avg 11.4 / 15.1 / 17.6, actual game
   lengths 12 / 15 / 17 plies in a random-vs-random sample).
2. **Throughput, even with zero move-ordering** and full `state.clone()`
   overhead, measured directly via plain Python negamax over real pyspiel
   state:

   | board | nodes/sec | depth 4 | depth 6 | depth 8 |
   |---|---|---|---|---|
   | 4×5 | ~330k | 0.04s | 0.6s | 9.7s |
   | 5×5 | ~300k | 0.07s | 2.5s | 23s |
   | 5×6 | ~285k | 0.12s | 4.9s | (untested, >3s budget) |

   These numbers are from the **opening position** — the worst case you'll
   ever face. `setup_initial_state` already burns 2–6 random plies before
   the teacher's first real decision, and branching collapses further as
   pieces get captured, so real mid/late-game searches will be cheaper than
   this table suggests; by the second half of the game it's often feasible
   to search straight to a terminal state.

**Recommendation**: iterative deepening with a time budget (same pattern
`othello_minimax.py` already uses — `time_budget`/`max_depth` exposed as
params, randomized per game), not a fixed depth, since branching varies so
much across one game. Add basic move ordering (even a cheap 1-ply heuristic
sort) — that alone typically buys 1.5–3x effective depth for the same time,
so the depth-8 numbers above are a floor, not a ceiling. Starting point:
`time_budget≈0.3–1.0s`, `max_depth≈14–16`. This is offline batch generation
parallelized across ~30 CPU cores, not latency-bound eval inference, so this
budget is comfortably affordable.

## V1 implementation plan

Mirror `othello_trajectories.py`'s skeleton exactly — Clobber is mechanically
almost identical (deterministic, strict turn alternation, no chance nodes).
Deliberately simple: no subgame decomposition, no group-count heuristic, no
dead-zone term, no double-view (teacher + opponent) recording — just
mobility-difference minimax teacher + MCTS opponent. Add the deferred ideas
later if experiments against this baseline justify them.

### 1. `envs/pvp_game_engine.py` — port `ClobberAgent`/`ClobberParams`

Port from `core/pvp/agents.py` / `core/models/pvp_models.py` in the eval repo:

```python
class ClobberParams(GameParams):
    rows: int
    columns: int

_CLOBBER_BOARD_SIZES = ((4, 5), (5, 5), (5, 6))
_CLOBBER_OPENING_PLIES = (2, 6)

class ClobberAgent(BaseGameAgent):
    def generate_params(self, config_id):
        rows, columns = _CLOBBER_BOARD_SIZES[config_id % len(_CLOBBER_BOARD_SIZES)]
        return ClobberParams(rows=rows, columns=columns)

    def format_state(self, state, player_id):
        colour = "o (White)" if player_id == 0 else "x (Black)"
        return f"You play {colour}.\n{state.observation_string(player_id)}"

    def setup_initial_state(self, state, seed):
        # reuse the shared random-opening helper already used for Othello
        _apply_seeded_random_opening(state, seed, _CLOBBER_OPENING_PLIES)
```

Register in `AGENT_REGISTRY["clobber"]`.

### 2. `envs/clobber_minimax.py` (new)

Negamax + alpha-beta directly over `pyspiel.State` — no text parsing needed
(unlike `othello_minimax.py`, there's no legacy HTTP-server text format to
match here, since this is a brand-new env built straight against pyspiel):

```python
def choose_action(state, player_id, time_budget, max_depth) -> int:
    # iterative deepening negamax/alpha-beta, time-budgeted
    # leaf eval: len(state.legal_actions(cur)) - len(state.legal_actions(opp))
```

### 3. `envs/clobber_trajectories.py` (new)

Copy `othello_trajectories.py`'s structure, single-view:

```python
def generate_heuristic_episode(game_id, max_turn=40) -> tuple[list[dict], float]:
    # rng = random.Random(game_id); teacher_seat = rng.choice((0, 1))
    # game = _AGENT.load_game(_AGENT.generate_params(config_id_for_task_id(game_id)))
    # state = game.new_initial_state(); _AGENT.setup_initial_state(state, seed=game_id)
    # opponent_bot = make_mcts_bot(...)
    # loop: teacher's turn -> clobber_minimax.choose_action
    #       opponent's turn -> mcts_step_or_none (truncate-on-search-failure,
    #       same handling already added for othello/leduc_poker/gin_rummy)
    # final_reward = score_for_player(state, teacher_seat) if terminal else 0.0
```

`max_turn` should be generous — game length is hard-capped at `cells - 1`
(≤29), so something like 40 effectively never truncates, meaning Clobber's
`final_reward` is almost always a real terminal outcome (a property the
other deterministic envs don't fully have).

### 4. Wiring

- `envs/sft_env_configs.py`: add `"clobber": generate_heuristic_episode` to
  `_SFT_REGISTRY`.
- `envs/generate_trajectories.py`: add `"clobber"` to `_FLAT_EXAMPLE_ENVS`,
  and an entry in `_ENV_DEFAULT_ARGS` —
  `{"num_games": ..., "max_turn": 40, "sample_by_score": True, "score_power": 1.0}`
  (mirrors Othello's score-filtering setup).
- `envs/shared_env.py`: `GAMES_TO_TASK_ID_RANGE["clobber"]` is **already
  present** (`700_000_000`–`799_999_999`) — no change needed.

### 5. Validation before trusting it

Same playbook as the other 5 envs: run a few hundred games locally, check
the score distribution isn't degenerate (teacher should clearly beat MCTS
more than half the time, but not ~100% — if so, dial back MCTS simulations
or the teacher's depth/budget for variety), and specifically check whether
Clobber's `RandomRolloutEvaluator` rollouts can hit the same "non-terminal
state with zero legal actions" quirk already found and fixed for gin_rummy
— wouldn't be surprising in a less battle-tested OpenSpiel game
implementation.

## Deferred to v2 (pending experiment results against the v1 baseline)

- Group-count minimization as a secondary heuristic term (Lorvi's verified
  real strategy).
- Dead-zone / potential-mobility lookahead term (vs. plain current-ply
  mobility difference).
- Component-restricted endgame search (the real subgame-decomposition win)
  if profiling shows depth is the bottleneck once the board fragments.
- Double-view recording (teacher + MCTS-opponent perspectives both become
  training examples, as Othello already does) if data yield needs boosting.
