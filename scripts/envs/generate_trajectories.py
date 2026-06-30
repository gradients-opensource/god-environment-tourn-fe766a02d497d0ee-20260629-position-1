"""
Generate game trajectories against env servers and save as an HF DatasetDict
(train / validation splits) ready for train_sft_env.py.

Analogous to tokenize_instruct.py but for environment SFT tasks.

Run from /workspace/scripts/:
  # Single environment (--num_games / --max_turn override the built-in defaults)
  python -m envs.generate_trajectories --environment_names liars_dice \
      --output_path /path/to/dataset --num_games 50000

  # Multiple environments — uses built-in per-env defaults for num_games/max_turn,
  # generates each to a staging path then merges into output_path
  python -m envs.generate_trajectories \
      --environment_names gin_rummy liars_dice leduc_poker \
      --output_path /path/to/dataset

  # Mixed: intercode (offline, reads from MINER_DATASETS) + game envs (env server)
  python -m envs.generate_trajectories \
      --environment_names gin_rummy intercode \
      --output_path /path/to/dataset

Score-based sampling:
  Some generators (e.g. leduc_poker) return (messages, score) tuples.  When
  --sample-by-score is set, each game is kept with probability
  clamp(score, 0, 1) ** score_power.  --wins-only is a stricter filter that
  discards any game where score <= 0.  For generators that return only
  messages (no score), all games are kept regardless of these flags.
"""

import argparse
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import Dataset, DatasetDict, concatenate_datasets

from envs.shared_env import GAMES_TO_TASK_ID_RANGE, _log
from envs.sft_env_configs import _OFFLINE_ENVS, get_sft_trajectory_generator


# ── Process-pool worker ───────────────────────────────────────────────────────
# Each worker process loads the expert generator once via _worker_init, then
# handles multiple games sequentially. Using processes (not threads) gives each
# worker its own GIL so CPU-bound expert/search/MCTS-opponent computation runs
# truly in parallel without contention. All 5 game generators now run entirely
# in-process (real pyspiel state, no env server -- see
# docs/PYSPIEL_NATIVE_DATAGEN_PLAN.md), so --num_workers tunes CPU
# parallelism, not a shared network resource.

_GENERATE_FN = None


def _worker_init(env_name: str) -> None:
    global _GENERATE_FN
    _GENERATE_FN = get_sft_trajectory_generator(env_name)


def _worker_play(game_id: int, max_turn: int) -> "list[dict] | tuple[list[dict], float] | None":
    return _GENERATE_FN(game_id, max_turn)

# ─────────────────────────────────────────────────────────────────────────────

MIN_ASSISTANT_TURNS = 1

# Envs whose SFT generator returns a list of already-flattened, stateless
# per-turn examples (``{"messages": [...], "tools": ...}``, §3.3/§3.4) instead
# of one growing conversation per game. These skip _clean/_sliding_windows
# entirely — each item is already a complete row. A generator may instead
# return ``(examples, score)`` (e.g. leduc_poker) — see the score-filter
# handling in generate_for_env, which applies wins_only/sample_by_score to the
# whole game's examples at once before flattening.
_FLAT_EXAMPLE_ENVS: frozenset[str] = frozenset({"liars_dice", "gin_rummy", "leduc_poker", "othello", "goofspiel", "clobber"})


def _sliding_windows(conv: list[dict], window_turns: int, window_step: int) -> list[list[dict]]:
    """
    Split a conversation into overlapping sub-conversations.
    Each window: [system] + window_turns × (user, assistant) pairs.
    Short games (fewer than window_turns pairs) are kept as one window.
    """
    system = [m for m in conv if m["role"] == "system"]
    turns  = [m for m in conv if m["role"] != "system"]

    pairs = []
    i = 0
    while i + 1 < len(turns):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i], turns[i + 1]))
            i += 2
        else:
            i += 1

    if not pairs:
        return []

    windows = []
    for start in range(0, len(pairs), window_step):
        chunk = pairs[start : start + window_turns]
        if not chunk:
            break
        window_conv = system[:]
        for user_msg, asst_msg in chunk:
            window_conv.extend([user_msg, asst_msg])
        windows.append(window_conv)

    return windows


def _clean(messages: "list[dict] | None") -> "list[dict] | None":
    """Normalize a raw conversation, preserving tool_calls and content: None.

    Plain-text generators emit string content, which passes through unchanged.
    Tool-calling generators may emit content: None alongside tool_calls — both
    are kept as-is (no stringification, no dropping) so the row matches the
    eval-time message shape.
    """
    if not messages:
        return None
    cleaned = []
    for m in messages:
        content = m.get("content")
        if content is not None and not isinstance(content, str):
            content = str(content)
        out = {"role": m["role"], "content": content}
        if m.get("tool_calls") is not None:
            out["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id") is not None:
            out["tool_call_id"] = m["tool_call_id"]
        cleaned.append(out)
    messages = cleaned
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not messages:
        return None
    if sum(1 for m in messages if m["role"] == "assistant") < MIN_ASSISTANT_TURNS:
        return None
    return messages


def _clean_tool_example(example: "dict | None") -> "dict | None":
    """Validate one flattened per-turn example for _FLAT_EXAMPLE_ENVS.

    Mirrors _clean's role for these envs: keep tool_calls/tools/content as-is
    (no stringification), but fail loudly (return None, counted as skipped) if
    the example isn't shaped as one assistant message carrying exactly one
    game_action tool call (plus zero or more memory tool calls before/after
    it — see liar_dice_trajectories.py's working_memory_append rows,
    docs/SFT_ALIGNMENT_PLAN.md §5.1a) — a malformed example here is exactly
    the defect this restructuring is meant to eliminate from the training
    distribution.
    """
    if not example or "messages" not in example or "tools" not in example:
        return None
    messages = example["messages"]
    if len(messages) < 3 or messages[-1].get("role") != "assistant":
        return None
    tool_calls = messages[-1].get("tool_calls") or []
    game_action_calls = [tc for tc in tool_calls if tc.get("function", {}).get("name") == "game_action"]
    if len(game_action_calls) != 1:
        return None
    return example


def _stats(conversations: list[list[dict]]) -> dict:
    turn_counts = [sum(1 for m in c if m["role"] == "assistant") for c in conversations]
    return {
        "total": len(conversations),
        "avg_assistant_turns": round(sum(turn_counts) / len(turn_counts), 2),
        "turn_distribution": dict(sorted(Counter(turn_counts).items())),
    }


# Per-environment generation defaults. Applied for both single and multi-env paths;
# CLI flags (--num_games, --max_turn, --score-power) override when explicitly provided.
# mcts_simulations is no longer threaded through here -- each generator now
# owns its own in-process MCTS opponent and simulation-count range internally
# (e.g. othello_trajectories._MCTS_SIMS_MIN/_MAX) since there's no shared env
# server whose load it needs to coordinate with.
_ENV_DEFAULT_ARGS: dict[str, dict] = {
    "gin_rummy":   {"num_games": 4000,   "max_turn": 200},
    "liars_dice":  {"num_games": 50000,  "max_turn": 30},
    "leduc_poker": {"num_games": 15000,  "max_turn": 10},
    "othello":     {"num_games": 8000,   "max_turn": 70},
    "goofspiel":   {"num_games": 10000,  "max_turn": 15},
    "clobber":     {"num_games": 8000,   "max_turn": 40,
                    "sample_by_score": True, "score_power": 1.0},
}
_DEFAULT_ARGS: dict = {"num_games": 50000, "max_turn": 30}


def _generate_offline(env_name: str, output_path: str) -> "int | None":
    """Generate dataset for offline envs that don't use an env server.

    Returns the number of examples saved, or None if the source was
    unavailable (caller should skip this env rather than abort).
    """
    if env_name == "intercode":
        from envs.intercode_dataset import build_intercode_sft_dataset
        dd = build_intercode_sft_dataset()
        if dd is None:
            print(
                "[generate_trajectories] No data for env 'intercode', skipping it: "
                "intercode dataset unavailable (training will continue with other envs). "
                "To include it, ensure MINER_DATASETS contains "
                "gradients-io-tournaments--intercode_bigcode_combined_12k "
                "and MINER_DATASETS_DIR is set.",
                flush=True,
            )
            return None
        dd.save_to_disk(output_path)
        _log(f"Intercode dataset saved → {output_path} ({len(dd['train'])} examples)")
        return len(dd["train"])
    else:
        raise ValueError(f"Unknown offline env: {env_name!r}")


def merge_datasets(per_env_paths: list[str], output_path: str) -> None:
    """Concatenate per-environment DatasetDicts into one and save to output_path."""
    splits: dict[str, list] = {}
    for p in per_env_paths:
        for split_name, ds in DatasetDict.load_from_disk(p).items():
            splits.setdefault(split_name, []).append(ds)
    merged = DatasetDict({k: concatenate_datasets(v) for k, v in splits.items()})
    merged.save_to_disk(output_path)
    total = sum(len(ds) for ds in merged.values())
    print(f"[generate_trajectories] Merged {len(per_env_paths)} env datasets -> {output_path} "
          f"({total} total examples)", flush=True)


def generate_for_env(
    env_name: str,
    output_path: str,
    num_games: int,
    max_turn: int,
    window_turns: int = 10,
    window_step: int = 0,
    num_workers: int = 0,
    seed: int = 42,
    wins_only: bool = False,
    sample_by_score: bool = False,
    score_power: float = 1.0,
    time_limit_seconds: float | None = None,
) -> int:
    """Generate and save a trajectory dataset for a single environment.

    Returns the number of examples saved.
    """
    is_flat = env_name in _FLAT_EXAMPLE_ENVS

    if window_step == 0:
        window_step = window_turns // 2 or 1

    task_id_min, task_id_max = GAMES_TO_TASK_ID_RANGE[env_name]

    # No env server: every game generator runs in-process (real pyspiel
    # state + an in-process MCTS opponent, see
    # docs/PYSPIEL_NATIVE_DATAGEN_PLAN.md), so the worker count is bounded by
    # CPU cores, not by a fixed externally-provided sidecar pool. Capped at
    # num_games so a small batch doesn't spin up idle worker processes.
    num_workers = num_workers or min(os.cpu_count() or 4, num_games)

    # One _log call, not five: with multiple envs running as concurrent OS
    # processes sharing the same stdout, five separate writes -- even
    # flushed individually -- can still each land between another process's
    # writes, shuffling this banner's own lines apart. A single multi-line
    # string is one write, so it can only interleave with another process's
    # banner as a whole block, not line-by-line.
    _log(
        f"Environment  : {env_name}\n"
        f"Output       : {output_path}\n"
        f"Num games    : {num_games}\n"
        f"Window turns : {window_turns}  step {window_step}\n"
        f"Workers      : {num_workers}  (cpu_count={os.cpu_count()})\n"
    )

    random.seed(seed)
    game_ids = random.sample(range(task_id_min + 1, task_id_max), num_games)
    tasks = [(gid, max_turn) for gid in game_ids]

    use_score_filter = wins_only or sample_by_score
    _log(f"Playing {num_games} games..." + (f" (limit {time_limit_seconds:.0f}s)" if time_limit_seconds else ""))
    if use_score_filter:
        _log(f"Score filter: wins_only={wins_only}  sample_by_score={sample_by_score}"
              f"  score_power={score_power}")
    conversations: list[list[dict]] = []
    examples: list[dict] = []  # _FLAT_EXAMPLE_ENVS only — already-flattened per-turn rows
    skipped = 0
    score_filtered = 0
    all_scores: list[float] = []
    # Othello-specific: view index 1 is the "opponent" (MCTS bot) view —
    # othello_trajectories.py hard-gates it to MCTS-outright-win games only
    # (see generate_heuristic_episode), so this tracks how many of the games
    # actually contributed a bot-view example vs. how many were played.
    bot_view_games_total = 0
    bot_view_games_kept = 0
    deadline = time.monotonic() + time_limit_seconds if time_limit_seconds else None
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(env_name,),
    ) as pool:
        futures = {pool.submit(_worker_play, gid, mt): gid for gid, mt in tasks}
        completed = 0
        for future in as_completed(futures):
            result = future.result()

            if is_flat:
                # result is list[dict] of already-flattened {"messages","tools"} rows,
                # (examples, score) when the generator also reports a score
                # (e.g. leduc_poker), or a list of (examples, score) "views"
                # when the generator produces multiple independently-scored
                # sample sets per game (e.g. othello's "ours" + "opponent"
                # double view) — score filters apply per view, before
                # flattening.
                if isinstance(result, tuple):
                    views = [result]
                elif result and isinstance(result[0], tuple):
                    views = result
                else:
                    views = [(result, None)]

                for view_idx, (flat_examples, score) in enumerate(views):
                    if score is not None:
                        all_scores.append(score)

                    if score is not None and use_score_filter:
                        if wins_only and score <= 0:
                            score_filtered += 1
                            flat_examples = []
                        elif sample_by_score:
                            prob = max(0.0, min(1.0, score)) ** score_power
                            if random.random() >= prob:
                                score_filtered += 1
                                flat_examples = []

                    if env_name == "othello" and view_idx == 1:
                        bot_view_games_total += 1
                        if flat_examples:
                            bot_view_games_kept += 1

                    for ex in (flat_examples or []):
                        cleaned = _clean_tool_example(ex)
                        if cleaned is None:
                            skipped += 1
                        else:
                            examples.append(cleaned)
                completed += 1
                if completed % 100 == 0:
                    _log(f"  {completed}/{num_games} games done", flush=True)
                if deadline is not None and time.monotonic() >= deadline:
                    _log(f"Time limit reached, stopping at {completed}/{num_games} games")
                    for f in futures:
                        f.cancel()
                    break
                continue

            # Unpack score when the generator returns (messages, score)
            if isinstance(result, tuple):
                raw_messages, score = result
                all_scores.append(score)
            else:
                raw_messages, score = result, None

            # Apply score-based filters only when a score is available
            if score is not None and use_score_filter:
                if wins_only and score <= 0:
                    score_filtered += 1
                    completed += 1
                    continue
                if sample_by_score:
                    prob = max(0.0, min(1.0, score)) ** score_power
                    if random.random() >= prob:
                        score_filtered += 1
                        completed += 1
                        continue

            cleaned = _clean(raw_messages)
            if cleaned is None:
                skipped += 1
            else:
                conversations.append(cleaned)
            completed += 1
            if completed % 100 == 0:
                _log(f"  {completed}/{num_games} games done", flush=True)

            if deadline is not None and time.monotonic() >= deadline:
                _log(f"Time limit reached, stopping at {completed}/{num_games} games")
                for f in futures:
                    f.cancel()
                break

    if is_flat:
        if use_score_filter:
            _log(f"Valid examples : {len(examples)}   Skipped : {skipped}   Score-filtered games : {score_filtered}")
        else:
            _log(f"Valid examples : {len(examples)}   Skipped : {skipped}")
        if env_name == "othello" and bot_view_games_total:
            pct = 100 * bot_view_games_kept / bot_view_games_total
            _log(f"Bot (opponent/MCTS) view: kept {bot_view_games_kept}/{bot_view_games_total} games "
                 f"({pct:.1f}%) — i.e. MCTS won outright that often")
        if not examples:
            raise RuntimeError("No valid examples generated.")
        dataset = Dataset.from_list(examples)
        dd = DatasetDict({"train": dataset})
        _log(f"Train: {len(dd['train'])}")
        dd.save_to_disk(output_path)
        _log(f"Dataset saved → {output_path}")
        return len(dd["train"])

    score_summary = ""
    if all_scores:
        wins = sum(1 for s in all_scores if s > 0)
        score_summary = (
            f"   Score stats: min={min(all_scores):.3f}  max={max(all_scores):.3f}"
            f"  wins(>0)={wins}/{len(all_scores)} ({100*wins/len(all_scores):.1f}%)\n"
            f"   Score-filtered: {score_filtered}"
        )
    _log(f"Valid : {len(conversations)}   Skipped : {skipped}{chr(10) + score_summary if score_summary else ''}")

    if not conversations:
        raise RuntimeError("No valid conversations generated.")

    # Raw game length stats — helps diagnose max_turn being hit or unexpectedly long games.
    raw_lengths = [sum(1 for m in c if m["role"] == "assistant") for c in conversations]
    max_turn_hits = sum(1 for l in raw_lengths if l >= max_turn)
    length_buckets = Counter(l // 10 * 10 for l in raw_lengths)
    # One _log call for the whole block (see the startup banner above for why).
    _log(
        f"\n  ── Raw game stats (before windowing) ──────────────────\n"
        f"  Avg turns/game   : {sum(raw_lengths)/len(raw_lengths):.1f}\n"
        f"  Min/Max turns    : {min(raw_lengths)} / {max(raw_lengths)}\n"
        f"  Hit max_turn={max_turn}  : {max_turn_hits} / {len(conversations)} games"
        f"  ({100*max_turn_hits/len(conversations):.1f}%)\n"
        f"  Length buckets   : " +
        "  ".join(f"{k}-{k+9}:{v}" for k, v in sorted(length_buckets.items()))
    )

    # Apply sliding window — expands long games into overlapping sub-conversations.
    # Short games (< window_turns pairs) are kept whole as a single window.
    windowed: list[list[dict]] = []
    for conv in conversations:
        windows = _sliding_windows(conv, window_turns, window_step)
        windowed.extend(windows if windows else [conv])
    conversations = windowed
    stats_lines = "\n".join(f"  {k}: {v}" for k, v in _stats(conversations).items())
    _log(
        f"\n  ── After windowing (turns={window_turns} step={window_step}) ──\n"
        f"  Total examples   : {len(conversations)}\n"
        f"{stats_lines}"
    )

    dataset = Dataset.from_list([{"messages": c} for c in conversations])
    dd = DatasetDict({"train": dataset})
    _log(f"Train: {len(dd['train'])}")

    dd.save_to_disk(output_path)
    _log(f"Dataset saved → {output_path}")
    return len(dd["train"])


def _run_one_env(
    env_name: str,
    output_path: str,
    *,
    is_offline: bool,
    num_games: int = 0,
    max_turn: int = 0,
    window_turns: int = 10,
    window_step: int = 0,
    num_workers: int = 0,
    seed: int = 42,
    wins_only: bool = False,
    sample_by_score: bool = False,
    score_power: float = 1.0,
    time_limit_seconds: "float | None" = None,
) -> "tuple[str, int] | None":
    """Run one env's full generation in its own OS process (main()'s
    multi-env path submits this to a ProcessPoolExecutor, so it must be a
    top-level function with plain picklable arguments, not a closure).

    Returns (output_path, example_count), or None if an offline env's source
    data was unavailable (caller treats that as skip-not-error, unlike a
    raised exception which means this env failed). Prints the count itself
    (plain print, not the SFT_ENV_VERBOSE-gated _log) so it always shows up
    even when this runs in a separate process via ProcessPoolExecutor.
    """
    if is_offline:
        count = _generate_offline(env_name, output_path)
        if count is None:
            return None
    else:
        count = generate_for_env(
            env_name, output_path,
            num_games=num_games,
            max_turn=max_turn,
            window_turns=window_turns,
            window_step=window_step,
            num_workers=num_workers,
            seed=seed,
            wins_only=wins_only,
            sample_by_score=sample_by_score,
            score_power=score_power,
            time_limit_seconds=time_limit_seconds,
        )
    print(f"[generate_trajectories] {env_name}: generated {count} examples -> {output_path}", flush=True)
    return output_path, count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--environment_names", nargs="+", required=True,
                   help="One or more environment names. Single entry uses --num_games / "
                        "--max_turn overrides; multiple entries use built-in per-env defaults.")
    p.add_argument("--output_path",      required=True)
    p.add_argument("--num_games",   type=int, default=None,
                   help="Override num_games (single-env only; defaults to per-env built-in).")
    p.add_argument("--max_turn",    type=int, default=None,
                   help="Override max_turn (single-env only; defaults to per-env built-in).")
    p.add_argument("--window_turns", type=int, default=10,
                   help="Split each game into sub-conversations of this many (user,assistant) "
                        "pairs. Games shorter than this are kept whole. Default 10.")
    p.add_argument("--window_step", type=int, default=0,
                   help="Slide window by this many pairs (default: window_turns // 2).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="Number of worker processes. Default 0 = min(os.cpu_count(), num_games) "
                        "(single-env) or os.cpu_count() split evenly across envs (multi-env). "
                        "Every game generator runs in-process (no env server), so this tunes "
                        "CPU parallelism.")
    p.add_argument("--seed", type=int, default=42)
    # Score-based sampling (for generators that return (messages, score) tuples)
    p.add_argument("--wins-only", action="store_true",
                   help="Discard games where score <= 0. Only applies when the "
                        "generator returns a (messages, score) tuple.")
    p.add_argument("--sample-by-score", action="store_true",
                   help="Keep each game with probability clamp(score, 0, 1) ** score-power. "
                        "Only applies when the generator returns a (messages, score) tuple.")
    p.add_argument("--score-power", type=float, default=None,
                   help="Exponent for score-based sampling (single-env override; "
                        "defaults to per-env built-in, or 1.0 if not set).")
    p.add_argument("--time_limit_seconds", type=float, default=None,
                   help="Total generation budget in seconds. For multiple envs the budget "
                        "is divided equally. None = unlimited (generate all num_games).")
    args = p.parse_args()

    if len(args.environment_names) == 1:
        env_name = args.environment_names[0]
        if env_name in _OFFLINE_ENVS:
            # Offline env (e.g. intercode): no env server, reads from MINER_DATASETS.
            # If data is unavailable, exit cleanly — nothing else to fall back to.
            count = _generate_offline(env_name, args.output_path)
            if count is None:
                _log(f"[generate_trajectories] No data for sole env {env_name!r}; nothing to train on.")
                return
            print(f"[generate_trajectories] {env_name}: generated {count} examples -> {args.output_path}", flush=True)
        else:
            # Game env: built-in per-env defaults apply; CLI flags override when provided
            env_cfg = {**_DEFAULT_ARGS, **_ENV_DEFAULT_ARGS.get(env_name, {})}
            count = generate_for_env(
                env_name, args.output_path,
                num_games=args.num_games if args.num_games is not None else env_cfg["num_games"],
                max_turn=args.max_turn if args.max_turn is not None else env_cfg["max_turn"],
                window_turns=args.window_turns,
                window_step=args.window_step,
                num_workers=args.num_workers,
                seed=args.seed,
                wins_only=args.wins_only or env_cfg.get("wins_only", False),
                sample_by_score=args.sample_by_score or env_cfg.get("sample_by_score", False),
                score_power=args.score_power if args.score_power is not None else env_cfg.get("score_power", 1.0),
                time_limit_seconds=args.time_limit_seconds,
            )
            print(f"[generate_trajectories] {env_name}: generated {count} examples -> {args.output_path}", flush=True)
    else:
        # Multiple envs: generate each to a staging path CONCURRENTLY, each
        # in its own OS process, then merge. Every game generator is
        # CPU-bound with no shared external resource (no env server), so
        # running them concurrently is a straightforward win: each gets its
        # own slice of the machine's cores instead of competing for a fixed
        # sidecar pool.
        #
        # --time_limit_seconds is NOT divided across envs (unlike the old
        # sequential code, where dividing was necessary -- env 1 running
        # sequentially could otherwise eat the whole budget and starve
        # everyone after it). Now every env starts together and
        # generate_for_env computes its own deadline as `now +
        # time_limit_seconds` relative to its own start, so giving each env
        # the full budget makes the whole concurrent job finish within
        # time_limit_seconds, not N times that. It's just an early-stopping
        # cap (checked once per completed game), so a fast env finishing
        # before its deadline is unaffected either way.
        game_env_names = [e for e in args.environment_names if e not in _OFFLINE_ENVS]
        per_env_limit = args.time_limit_seconds

        # Split the machine's CPU cores evenly across the envs running
        # concurrently. Explicit --num_workers overrides this for every env.
        if args.num_workers:
            per_env_num_workers = {e: args.num_workers for e in game_env_names}
        else:
            cpu_total = os.cpu_count() or 4
            shared = max(1, cpu_total // len(game_env_names)) if game_env_names else cpu_total
            per_env_num_workers = {e: shared for e in game_env_names}

        per_env_paths_by_env: dict[str, str] = {}
        per_env_counts: dict[str, int] = {}
        with ProcessPoolExecutor(max_workers=len(args.environment_names)) as pool:
            futures = {}
            for env_name in args.environment_names:
                env_path = f"{args.output_path}_{env_name}"
                if env_name in _OFFLINE_ENVS:
                    future = pool.submit(_run_one_env, env_name, env_path, is_offline=True)
                else:
                    env_cfg = {**_DEFAULT_ARGS, **_ENV_DEFAULT_ARGS.get(env_name, {})}
                    future = pool.submit(
                        _run_one_env, env_name, env_path,
                        is_offline=False,
                        num_games=env_cfg["num_games"],
                        max_turn=env_cfg["max_turn"],
                        window_turns=args.window_turns,
                        window_step=args.window_step,
                        num_workers=per_env_num_workers.get(env_name, 0),
                        seed=args.seed,
                        wins_only=env_cfg.get("wins_only", False),
                        sample_by_score=env_cfg.get("sample_by_score", False),
                        score_power=env_cfg.get("score_power", 1.0),
                        time_limit_seconds=per_env_limit,
                    )
                futures[future] = env_name

            for future in as_completed(futures):
                env_name = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        env_path, count = result
                        per_env_paths_by_env[env_name] = env_path
                        per_env_counts[env_name] = count
                    # else: offline env unavailable, skip silently — warning
                    # already printed inside _generate_offline
                except Exception as exc:
                    print(f"[generate_trajectories] No data for env {env_name!r}, skipping it: {exc}", flush=True)
                    # else: skip this env and continue with the rest

        # Per-env counts were already printed by _run_one_env as each one
        # finished (so they show up live, not just at the very end); this is
        # the always-shown recap of all of them together once the whole
        # multi-env job is done.
        print("[generate_trajectories] Samples generated per env:", flush=True)
        for env_name in args.environment_names:
            print(f"  {env_name}: {per_env_counts.get(env_name, 0)}", flush=True)

        if not per_env_paths_by_env:
            raise RuntimeError(
                "No valid data generated for any environment "
                f"({', '.join(args.environment_names)}). Check MINER_DATASETS "
                "(for intercode) and the per-env error messages above."
            )
        # Deterministic merge order regardless of which env finished first.
        per_env_paths = [per_env_paths_by_env[e] for e in args.environment_names if e in per_env_paths_by_env]
        merge_datasets(per_env_paths, args.output_path)


if __name__ == "__main__":
    main()
