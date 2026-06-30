"""Trajectory generator for Goofspiel SFT training.

Produces per-turn tool-calling examples (§3.2/§3.3 of docs/SFT_ALIGNMENT_PLAN.md):
each turn of a game becomes its own standalone
``{"messages": [system, user, assistant(tool_calls=[...,game_action])], "tools": ...}``
example — a stateless [system, user] reconstruction matching
``core/pvp/bot.py::LLMBot._run_turn`` at eval time, with all 5 tools (4 memory
+ game_action) offered every turn. The memory block is empty until the
opponent-pattern detection below fires; see "Memory-tool use."

Game mechanics now run on a real
``pyspiel.convert_to_turn_based(pyspiel.load_game("goofspiel", ...))`` state
(see docs/PYSPIEL_NATIVE_DATAGEN_PLAN.md), replacing the from-scratch local
card simulator this file used previously. That simulator was built after the
mcts-api env server was found to be flatly wrong for this game (see
~/sn56-G.O.D-env/docs/GOOFSPIEL_IMPLEMENTATION_NOTES.md and
GOOFSPIEL_LOCAL_ROLLOUT_SPEC.md: it flattened both players' simultaneous
actions into a single int via the wrong pyspiel API, corrupting every round
after the first) — pyspiel itself was never the problem, only the server's
HTTP wrapper around it was, so driving pyspiel directly (no server in
between) removes the bug at its root rather than routing around it. The
real engine's turn-based view of this simultaneous-move game only exposes
the *acting* player's legal actions at each decision (confirmed by direct
testing: ``legal_actions(other_player)`` is empty mid-round), so the
opponent's hand for the teacher's lookahead is still tracked locally exactly
as before (``hands[player]``, discarded as each side's card is observed) —
pyspiel doesn't expose it any other way for an imp_info=True game, and this
generator already had legitimate oracle access to it (it's the one driving
both sides). What changed: each round's prize card now comes from pyspiel's
own chance node (instead of an independently pre-shuffled local deck), and
both players' chosen cards are applied as real actions on a real state,
which can be sanity-checked against ``state.returns()`` (see §6 of the
plan). The rendered observation text and the teacher/opponent policy
functions below are UNCHANGED.

Policy: goof_spiel_minimax.choose_action — each round is solved as a
zero-sum matrix game over the (visible-to-the-teacher) hands, with a fast
heuristic estimate of the value of future rounds, and the deterministic mode
of the resulting equilibrium is played (see goof_spiel_minimax.py's
docstring for why the mode rather than a sample is used as the SFT label).

Opponent: one of three strategies, picked per game (seeded by game_id, so
reproducible) for variety in what the teacher has to beat:
  - "random": uniform-random legal bid -- matches what the old "mcts"
    request actually produced server-side.
  - "mirror": bids the card matching the revealed prize (nearest higher,
    then nearest lower, if already spent) -- disciplined but exploitable,
    the same heuristic this generator's policy replaced as a baseline.
  - "minimax": same zero-sum-matrix-game solve as the teacher's own policy,
    but SAMPLED from the equilibrium mixed strategy rather than taking the
    argmax -- unlike the teacher (which needs one deterministic label per
    state), this is a live repeated-play opponent, so mixing is the
    correct, harder-to-exploit choice.

Memory-tool use: unlike the other phase-1 generators (which never exercise
memory tools), this one does, because it's the natural fit -- each turn's
prompt is a stateless [system, user] reconstruction with no visibility into
earlier rounds, so a pattern noticed in round 2 is invisible again in round 3
unless something durable records it.

Two separate thresholds, deliberately not the same:
  - The teacher's own move choice (``choose_action``'s ``opp_bid_history``)
    uses ground-truth detection that can fire as early as round 1 -- it's
    allowed full oracle access for picking its own best move, same as the
    equilibrium solve always had. This maximizes win rate; it's invisible
    to the model (nothing about it appears in the rendered text).
  - The memory NOTE -- text the model itself is trained to write -- is
    gated separately by ``_MEMORY_NOTE_MIN_ROUND``: even once the ground
    truth confirms the pattern, the note isn't written until that many
    rounds have actually been played. A real model only has the weaker,
    hidden-bid-free signal (its own bid, the prize, who won) to work with,
    so it would plausibly take a few rounds to notice -- writing the note
    immediately would train an unrealistically instant "realization" that
    doesn't match what's actually inferrable from a handful of outcomes.
Once both conditions hold, that round's assistant response writes the note
via ``working_memory_rewrite`` *in the same response* as ``game_action``
(mirrors core/pvp/bot.py's ``_run_turn``: one model response may edit memory
and must call ``game_action``, not a separate round-trip) — every later
round's system prompt then renders that note as already-there context, same
as a real game.
"""

import json
import random

from envs.goof_spiel_minimax import _greedy_continuation, _predict_mirror_bid, _solve_zero_sum_game
from envs.goof_spiel_minimax import choose_action as _choose_lookahead_action
from envs.goof_spiel_minimax import detect_mirror_opponent
from envs.pvp_format import build_full_system_prompt
from envs.pvp_format import build_pvp_tools
from envs.pvp_format import build_user_prompt
from envs.pvp_format import default_memories
from envs.pvp_format import split_normalized_observation
from envs.pvp_format import tools_to_openai
from envs.pvp_game_engine import GoofspielAgent
from envs.pvp_game_engine import GoofspielParams
from envs.pvp_game_engine import score_for_player
from envs.pvp_models import MemoryArea
from envs.pvp_models import MemoryOp
from envs.pvp_tools import memory_tool_name
from envs.shared_env import _log

_AGENT = GoofspielAgent()

_GAME_NAME = "goofspiel"
_WORKING_REWRITE_TOOL = memory_tool_name(MemoryArea.WORKING, MemoryOp.REWRITE)
_MIRROR_NOTE_SLOT = 1
# 0-indexed round at which the memory note may first be written, even if
# ground truth confirms the pattern earlier -- see module docstring.
_MEMORY_NOTE_MIN_ROUND = 4

# Matches core/pvp/agents.py::GoofspielAgent.generate_params()'s diversity.
_NUM_CARDS_CHOICES = (5, 8, 10, 13)
_OPPONENT_STRATEGIES = ("random", "mirror", "minimax")

# rules + empty memory block + tool guidance, mirrors LLMBot._system_prompt
# with a fresh (all-empty) memory state — see §3.2.
_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)


# ---------------------------------------------------------------------------
# Opponent strategies — my_hand/opp_hand/future are all from ground-truth
# simulator state, not parsed off rendered text.
# ---------------------------------------------------------------------------

def _bot_random(my_hand: "list[int]", opp_hand: "list[int]", prize: int, future: "list[int]", rng: random.Random) -> int:
    return rng.choice(my_hand)


def _bot_mirror(my_hand: "list[int]", opp_hand: "list[int]", prize: int, future: "list[int]", rng: random.Random) -> int:
    return _predict_mirror_bid(my_hand, prize)


def _bot_minimax(my_hand: "list[int]", opp_hand: "list[int]", prize: int, future: "list[int]", rng: random.Random) -> int:
    me = sorted(my_hand)
    if len(me) == 1:
        return me[0]
    opp = sorted(opp_hand)
    M = [
        [
            (prize if i > j else (-prize if i < j else 0.0))
            + _greedy_continuation([c for c in me if c != i], [c for c in opp if c != j], future)
            for j in opp
        ]
        for i in me
    ]
    _, x = _solve_zero_sum_game(M)
    return rng.choices(me, weights=x)[0]


_BOTS = {"random": _bot_random, "mirror": _bot_mirror, "minimax": _bot_minimax}


# ---------------------------------------------------------------------------
# Observation rendering — matches split_normalized_observation's contract
# (envs/pvp_format.py): a "You are Player N." line before a "Legal Actions:"
# header, action lines "K -> label". Only the mover's own hand is shown, to
# match real eval-time (imp_info=True) observations.
# ---------------------------------------------------------------------------

def _render_observation(
    player_id: int, prize: int, future: "list[int]", scores: "list[int]",
    my_hand: "list[int]", win_sequence: "list[int]",
) -> str:
    state_lines = [
        "Current State:",
        f"Current point card: {prize}",
        f"Remaining Point Cards: {''.join(str(c) for c in sorted(future))}",
        f"Points: {scores[0]} {scores[1]}",
        f"P{player_id} hand: " + " ".join(str(c) for c in sorted(my_hand)),
        "Win sequence: " + " ".join(str(w) for w in win_sequence),
    ]
    legal_lines = [f"{c - 1} -> [P{player_id}]Bid: {c}" for c in sorted(my_hand)]
    return (
        "\n".join(state_lines)
        + f"\n\nYou are Player {player_id}.\nLegal Actions:\n"
        + "\n".join(legal_lines)
        + "\n\nYour choice (ID only):"
    )


def _build_tool_example(
    observation: str, action_id: int, system_prompt: str = _SYSTEM_PROMPT, memory_note: "str | None" = None,
) -> dict:
    """Build one stateless {system, user, assistant(tool_calls)} training example.

    ``system_prompt`` defaults to the always-empty-memory block; pass one
    from ``build_full_system_prompt(_GAME_NAME, memories=...)`` to render a
    non-empty memory state (see ``generate_heuristic_episode``). When
    ``memory_note`` is given, the assistant's response writes it to the
    working-memory slot in the SAME tool_calls list as game_action, mirroring
    core/pvp/bot.py's "one response, optionally edit memory, then commit a
    move" turn shape — not a separate turn.
    """
    state_desc, player_id, legal_actions = split_normalized_observation(observation)
    tools = build_pvp_tools([aid for aid, _ in legal_actions])
    tool_calls = []
    if memory_note is not None:
        tool_calls.append({
            "type": "function",
            "function": {
                "name": _WORKING_REWRITE_TOOL,
                "arguments": json.dumps({"slot": _MIRROR_NOTE_SLOT, "content": memory_note}),
            },
        })
    tool_calls.append({"type": "function", "function": {"name": "game_action", "arguments": json.dumps({"action_id": action_id})}})
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(state_desc, player_id, legal_actions)},
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
        ],
        # JSON-encoded: see intercode_dataset.py's _INTERCODE_TOOLS_JSON for why
        # heterogeneous tool `parameters.properties` must not go through
        # Dataset.from_list as a list-of-dicts column.
        "tools": json.dumps(tools_to_openai(tools)),
    }


def generate_heuristic_episode(
    game_id: int,
    max_turn: int = 15,
) -> "tuple[list[dict], float]":
    """
    Run one Goofspiel game on a real pyspiel state using goof_spiel_minimax's
    lookahead policy against a randomly-chosen (per game_id) opponent
    strategy.

    Returns ``(examples, final_reward)``, where ``examples`` is a list of
    independent per-turn training examples (one per round played), each shaped
    ``{"messages": [system, user, assistant(tool_calls=[game_action])], "tools": [...]}``.
    ``final_reward`` is the normalized terminal score in [0, 1] (0.0 = loss,
    0.5 = draw, 1.0 = win — sign-only, matching ``returns_type="win_loss"``).
    Read directly from pyspiel's own ``state.returns()`` whenever the game
    actually reaches a terminal state — NOT re-derived from the local
    scores tally below, which only covers the rounds this loop actually
    drove. pyspiel auto-resolves the LAST round internally once both hands
    are down to exactly one card (the outcome is fully forced, no real
    decision left) without offering a chance node or bid actions for it
    (confirmed live: terminal fires after exactly num_cards-1 explicit
    rounds, for every deck size) — local tracking is missing that round's
    point award, pyspiel's returns() isn't. Falls back to the local
    win/loss/draw comparison only when ``max_turn`` cut the game off before
    it reached a real terminal state.
    """
    rng = random.Random(game_id)
    num_cards = rng.choice(_NUM_CARDS_CHOICES)
    player_id = rng.choice((0, 1))
    opp_id = 1 - player_id
    bot_fn = _BOTS[rng.choice(_OPPONENT_STRATEGIES)]

    game = _AGENT.load_game(GoofspielParams(
        players=2, num_cards=num_cards, imp_info=True,
        points_order="random", returns_type="win_loss",
    ))
    state = game.new_initial_state()

    try:
        # Oracle hand tracking for the teacher's lookahead (see module
        # docstring: pyspiel's turn-based view of this imp_info=True game
        # only exposes the acting player's own legal actions, so the
        # opponent's hand is tracked locally exactly as before).
        hands = [set(range(1, num_cards + 1)), set(range(1, num_cards + 1))]
        scores = [0, 0]
        win_sequence: list[int] = []
        remaining_prizes = set(range(1, num_cards + 1))

        examples: list[dict] = []
        # (prize, opp_hand_before_round, opp_actual_bid) per round played --
        # ground truth, feeds choose_action's own move choice (oracle access,
        # can detect from round 1). NOT what gets shown to the model -- see
        # _MEMORY_NOTE_MIN_ROUND below for the separate, slower memory gate.
        opp_bid_history: "list[tuple[int, list[int], int]]" = []
        mirror_already_noted = False
        # Real shape (PVP_WORKING/LONGTERM_*), not a stub -- this gets rendered
        # into the prompt, so it must match what eval-time observations show.
        memories = default_memories()
        working_memory = memories[MemoryArea.WORKING]

        for round_idx in range(min(num_cards, max_turn)):
            if state.is_terminal():
                break

            # Resolve this round's prize-card reveal -- a real pyspiel chance
            # node (replaces the old independently pre-shuffled local deck).
            outcomes, probs = zip(*state.chance_outcomes())
            chosen = rng.choices(outcomes, weights=probs)[0]
            prize = chosen + 1
            state.apply_action(chosen)
            remaining_prizes.discard(prize)
            future = sorted(remaining_prizes)

            my_hand = sorted(hands[player_id])
            opp_hand = sorted(hands[opp_id])

            observation = _render_observation(player_id, prize, future, scores, my_hand, win_sequence)
            my_card = (
                _choose_lookahead_action(my_hand, opp_hand, prize, future, opp_bid_history)
                if len(my_hand) > 1 else my_hand[0]
            )

            # Ground truth may already confirm the pattern (as early as
            # round 1); the note still waits for _MEMORY_NOTE_MIN_ROUND, to
            # match how long it'd plausibly take a real model to notice from
            # its own weaker (hidden-bid-free) observations.
            just_detected = (
                not mirror_already_noted
                and round_idx >= _MEMORY_NOTE_MIN_ROUND
                and detect_mirror_opponent(opp_bid_history)
            )
            if just_detected:
                mirror_already_noted = True
                note = (
                    "Every round so far, the outcome has been exactly what you'd expect "
                    "if this opponent always bids the prize card itself: I win when I bid "
                    "higher than the prize, lose when I bid lower, tie when I bid the same. "
                    "Countering by bidding the cheapest card above the prize each round, "
                    "to win it cheaply and keep higher cards for rounds I can't win this way."
                )
                working_memory.rewrite(_MIRROR_NOTE_SLOT, note)
                system_prompt = build_full_system_prompt(_GAME_NAME, memories=memories)
                examples.append(_build_tool_example(observation, my_card - 1, system_prompt=system_prompt, memory_note=note))
            elif mirror_already_noted:
                system_prompt = build_full_system_prompt(_GAME_NAME, memories=memories)
                examples.append(_build_tool_example(observation, my_card - 1, system_prompt=system_prompt))
            else:
                examples.append(_build_tool_example(observation, my_card - 1))

            opp_card = bot_fn(opp_hand, my_hand, prize, future, rng)
            opp_bid_history.append((prize, opp_hand, opp_card))

            # pyspiel always lets player 0 act first within a round (verified
            # by direct testing) -- apply both real actions in that order
            # regardless of which seat is "ours".
            cards_by_seat = {player_id: my_card, opp_id: opp_card}
            first_actor = state.current_player()
            second_actor = 1 - first_actor
            state.apply_action(cards_by_seat[first_actor] - 1)
            state.apply_action(cards_by_seat[second_actor] - 1)

            if my_card > opp_card:
                scores[player_id] += prize
                win_sequence.append(player_id)
            elif opp_card > my_card:
                scores[opp_id] += prize
                win_sequence.append(opp_id)
            else:
                win_sequence.append(-1)

            hands[player_id].discard(my_card)
            hands[opp_id].discard(opp_card)
        else:
            _log(f"[goof_spiel_trajectories] max_turn={max_turn} reached (game {game_id})")

        if state.is_terminal():
            # Authoritative: pyspiel auto-resolves the LAST round internally
            # once both hands are down to exactly one card (the outcome is
            # fully forced -- one point card and one card per hand left, no
            # real decision) without offering an explicit chance node or bid
            # actions for it (confirmed live: terminal fires after exactly
            # num_cards-1 explicit rounds, for every deck size). The local
            # scores/win_sequence tally above only covers the rounds this
            # loop actually drove, so it's missing that forced final round's
            # point award -- state.returns() already includes it, so use it
            # directly instead of re-deriving (and risking the same gap)
            # from local bookkeeping.
            final_reward = score_for_player(state, player_id)
        else:
            diff = scores[player_id] - scores[opp_id]
            final_reward = 0.5 if diff == 0 else (1.0 if diff > 0 else 0.0)
        return examples, final_reward
    except Exception as exc:
        _log(f"[goof_spiel_trajectories] Failed to build episode (game {game_id}): {exc}")
        return [], 0.0
