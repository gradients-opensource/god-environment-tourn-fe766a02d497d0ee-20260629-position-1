"""Expert trajectory generator for Gin Rummy SFT training.

Produces per-turn tool-calling examples (§3.2/§3.3 of docs/SFT_ALIGNMENT_PLAN.md):
each turn of a game becomes its own standalone
``{"messages": [system, user, assistant(tool_calls=[...,game_action])], "tools": ...}``
example — a stateless [system, user] reconstruction matching
``core/pvp/bot.py::LLMBot._run_turn`` at eval time, with all 5 tools (4 memory
+ game_action) offered every turn. The memory block carries a running
opponent-signal note (see "Opponent tracking" below) once anything has been
observed; empty before that.

Game mechanics now run on a real ``pyspiel.load_game("gin_rummy", ...)``
state (see docs/PYSPIEL_NATIVE_DATAGEN_PLAN.md) instead of the HTTP env
server. The expert policy (``get_expert_action`` and its ``choose_*``
helpers below) is UNCHANGED — it parses a single rendered text blob via
regex, and pyspiel's real gin_rummy ``observation_string`` turned out to be
byte-identical to what the old server returned (confirmed by direct
testing) — the server was just a thin HTTP wrapper around this exact
pyspiel call. ``_build_observation_envelope`` reconstructs that same blob
shape directly from live state, so every downstream parser needed zero
changes.

Opponent tracking: every turn between two of our own decisions, the opponent
takes one full turn (draw + discard) that we never see directly — we only see
the discard pile before and after. Two things are inferable purely from that
delta, with no access to the opponent's actual hand:
  - If the card we ourselves just discarded is gone from the pile next time we
    look, the opponent took it — meaning it was useful enough to prefer over a
    blind stock draw, i.e. it likely extends a meld near that rank or suit.
  - Whatever new card appears on top of the pile is the opponent's own
    discard — a (weaker, but still real) signal that they don't urgently need
    that rank right now.
The naive way to compute this (diff the previous and current discard pile) is
wrong on our own Discard-phase turns, since our own discard also adds a new
card to the pile and would be misattributed to the opponent. ``_pile_signal``
explicitly subtracts our own last discard from the "expected" pile before
diffing, so only genuinely opponent-caused changes are reported (verified
against a real game trace: zero false positives/negatives across 16 turns).

This is exactly the kind of information a real eval-time player would need
memory for: the moment a pickup happens, it's directly inferable from two
consecutive observations, but it's never visible again afterward — the
current discard pile shows *that* a card exists, not *when in the game* or
*by whom* it was added. Each turn's note is a compact rolling summary (ranks/
suits the opponent has taken vs. discarded), rewritten via
``working_memory_rewrite`` whenever a new signal arrives — not appended,
since old individual cards lose their value as a stale, swelling log faster
than this game's other memory-eligible signals (contrast goofspiel/
liars_dice, where the full history stays relevant).
"""

import json
import random
import re
from collections import Counter, defaultdict
from typing import Optional

from envs.pvp_format import build_full_system_prompt
from envs.pvp_format import build_pvp_tools
from envs.pvp_format import build_user_prompt
from envs.pvp_format import default_memories
from envs.pvp_format import split_normalized_observation
from envs.pvp_format import tools_to_openai
from envs.pvp_game_engine import config_id_for_task_id
from envs.pvp_game_engine import GinRummyAgent
from envs.pvp_game_engine import make_mcts_bot
from envs.pvp_game_engine import mcts_step_or_none
from envs.pvp_models import MemoryArea
from envs.pvp_models import MemoryOp
from envs.pvp_tools import memory_tool_name
from envs.shared_env import _log

_MCTS_SIMS_MIN = 25
_MCTS_SIMS_MAX = 50
_GAME_NAME = "gin_rummy"
_AGENT = GinRummyAgent()

# rules + empty memory block + tool guidance, mirrors LLMBot._system_prompt
# with a fresh (all-empty) memory state — see §3.2.
_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)

_WORKING_NOTE_SLOT = 1
_WORKING_REWRITE_TOOL = memory_tool_name(MemoryArea.WORKING, MemoryOp.REWRITE)
# How many recent taken/discarded cards to summarize in the note — bounds
# memory token usage; older entries roll off (least decision-relevant).
_NOTE_HISTORY_LEN = 6

##############################################################################################
# Card utilities
##############################################################################################

CARD_VALUES = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 10, 'Q': 10, 'K': 10,
}
RANK_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
RANK_IDX   = {r: i for i, r in enumerate(RANK_ORDER)}


def card_value(card: str) -> int:
    return CARD_VALUES.get(card[0].upper(), 0) if len(card) >= 2 else 0


def get_rank(card: str) -> str:
    return card[0].upper()


def get_suit(card: str) -> str:
    return card[1].lower()


def find_all_melds(hand: list[str]) -> list[frozenset[str]]:
    melds: list[frozenset[str]] = []

    rank_groups: dict[str, list[str]] = defaultdict(list)
    for c in hand:
        rank_groups[get_rank(c)].append(c)
    for cards in rank_groups.values():
        if len(cards) >= 3:
            melds.append(frozenset(cards[:3]))
        if len(cards) >= 4:
            melds.append(frozenset(cards[:4]))

    suit_groups: dict[str, list[str]] = defaultdict(list)
    for c in hand:
        suit_groups[get_suit(c)].append(c)
    for cards in suit_groups.values():
        sorted_cards = sorted(cards, key=lambda c: RANK_IDX[get_rank(c)])
        i = 0
        while i < len(sorted_cards):
            run = [sorted_cards[i]]
            j = i + 1
            while j < len(sorted_cards):
                if RANK_IDX[get_rank(sorted_cards[j])] == RANK_IDX[get_rank(run[-1])] + 1:
                    run.append(sorted_cards[j])
                    j += 1
                else:
                    break
            for start in range(len(run)):
                for end in range(start + 3, len(run) + 1):
                    melds.append(frozenset(run[start:end]))
            i = j if len(run) > 1 else i + 1

    return melds


def _build_hand_state(hand: list[str]) -> tuple[int, list[int], list[int]]:
    n = len(hand)
    card_to_idx = {c: i for i, c in enumerate(hand)}
    meld_masks: list[int] = []
    for meld in find_all_melds(hand):
        mask, valid = 0, True
        for c in meld:
            if c not in card_to_idx:
                valid = False; break
            mask |= (1 << card_to_idx[c])
        if valid:
            meld_masks.append(mask)
    vals = [card_value(c) for c in hand]
    return n, vals, meld_masks


def compute_optimal_deadwood(hand: list[str]) -> int:
    if not hand:
        return 0
    n, vals, meld_masks = _build_hand_state(hand)
    memo: dict[int, int] = {}

    def _dp(used: int) -> int:
        if used in memo:
            return memo[used]
        best = sum(vals[i] for i in range(n) if not (used >> i & 1))
        for mm in meld_masks:
            if not (mm & used):
                best = min(best, _dp(used | mm))
        memo[used] = best
        return best

    return _dp(0)


def get_optimal_meld_cards(hand: list[str]) -> set[str]:
    if not hand:
        return set()
    n, vals, meld_masks = _build_hand_state(hand)
    memo: dict[int, tuple[int, int]] = {}

    def _dp(used: int) -> tuple[int, int]:
        if used in memo:
            return memo[used]
        best_dw    = sum(vals[i] for i in range(n) if not (used >> i & 1))
        best_final = used
        for mm in meld_masks:
            if not (mm & used):
                child_dw, child_final = _dp(used | mm)
                if child_dw < best_dw:
                    best_dw, best_final = child_dw, child_final
        memo[used] = (best_dw, best_final)
        return best_dw, best_final

    _, optimal_mask = _dp(0)
    return {hand[i] for i in range(n) if (optimal_mask >> i & 1)}


def meld_potential(upcard: str, hand: list[str]) -> int:
    if not upcard or upcard == 'XX' or len(upcard) != 2:
        return 0
    return max(0, compute_optimal_deadwood(hand) - compute_optimal_deadwood(hand + [upcard]))


def is_adjacent_in_suit(card: str, hand: list[str]) -> bool:
    idx  = RANK_IDX[get_rank(card)]
    suit = get_suit(card)
    for c in hand:
        if c == card or get_suit(c) != suit:
            continue
        if abs(RANK_IDX[get_rank(c)] - idx) <= 1:
            return True
    return False


def partial_run_can_complete(card: str, hand: list[str], dead: set[str]) -> bool:
    idx  = RANK_IDX[get_rank(card)]
    suit = get_suit(card)
    indices_in_hand = sorted(RANK_IDX[get_rank(c)] for c in hand if get_suit(c) == suit)
    if idx not in indices_in_hand:
        return True
    seg = [idx]
    for i in sorted(indices_in_hand):
        if i == seg[-1] + 1:
            seg.append(i)
        elif i > seg[-1] + 1 and seg[0] <= idx <= seg[-1]:
            break
        elif i < idx:
            if not seg or i == seg[0] - 1:
                seg.insert(0, i)
    if len(seg) >= 3:
        return True
    lo, hi = seg[0], seg[-1]
    if lo > 0 and (RANK_ORDER[lo - 1] + suit) not in dead:
        return True
    if hi < 12 and (RANK_ORDER[hi + 1] + suit) not in dead:
        return True
    return False


def partial_set_can_complete(card: str, hand: list[str], dead: set[str]) -> bool:
    rank      = get_rank(card)
    hand_suits = {get_suit(c) for c in hand if get_rank(c) == rank}
    if len(hand_suits) >= 3:
        return True
    for s in 'shdc':
        candidate = rank + s
        if s not in hand_suits and candidate not in dead:
            return True
    return False


##############################################################################################
# Observation parsers
##############################################################################################

_RE_PHASE        = re.compile(r'Phase:\s*(\w+)')
_RE_PLAYER       = re.compile(r'You are Player (\d+)')
_RE_UPCARD       = re.compile(r'Upcard:\s*(\w+)')
_RE_DEADWOOD     = re.compile(r'Deadwood=(\d+)')
_RE_KNOCK_CARD   = re.compile(r'Knock card:\s*(\d+)')
_RE_LEGAL        = re.compile(r'^\s*(\d+)\s*->\s*Player:\s*\d+\s*Action:\s*(.+)$', re.MULTILINE)
_RE_DISCARD_PILE = re.compile(r'Discard pile[:\s]+([^\n]+)', re.IGNORECASE)
_RE_CARD         = re.compile(r'([A2-9TJQK][shdc])')
_RE_CARD_EXACT   = re.compile(r'^([A2-9TJQK][shdc])$')
_RE_MELD_GROUP   = re.compile(r'^([A2-9TJQK][shdc]){2,}$')


def parse_phase(obs: str) -> str:
    m = _RE_PHASE.search(obs)
    return m.group(1) if m else ''


def parse_hand(obs: str) -> list[str]:
    player_match = _RE_PLAYER.search(obs)
    pid     = player_match.group(1) if player_match else '0'
    section = re.search(
        rf'Player{pid}: Deadwood=\d+.*?\n\+-+\+\n(.*?)\n\+-+\+',
        obs, re.DOTALL
    )
    if not section:
        return []
    cards = []
    for row in section.group(1).strip().split('\n'):
        cards.extend(_RE_CARD.findall(row))
    return cards


def parse_upcard(obs: str) -> str:
    m = _RE_UPCARD.search(obs)
    return m.group(1) if m else 'XX'


def parse_deadwood(obs: str) -> int:
    m = _RE_DEADWOOD.search(obs)
    return int(m.group(1)) if m else 99


def parse_knock_card(obs: str) -> int:
    m = _RE_KNOCK_CARD.search(obs)
    return int(m.group(1)) if m else 10


def parse_legal_actions(obs: str) -> list[tuple[str, str]]:
    return _RE_LEGAL.findall(obs)


def parse_discard_pile(obs: str, upcard: Optional[str] = None) -> set[str]:
    cards: set[str] = set()
    m = _RE_DISCARD_PILE.search(obs)
    if m:
        cards.update(_RE_CARD.findall(m.group(1)))
    if upcard is None:
        upcard = parse_upcard(obs)
    if upcard and upcard != 'XX' and len(upcard) == 2:
        cards.add(upcard)
    return cards


##############################################################################################
# Opponent tracking — see module docstring for why the diff must subtract our
# own last discard before attributing changes to the opponent.
##############################################################################################

def pile_signal(
    known_pile: "set[str]", my_last_discard: "str | None", pile: "set[str]",
) -> "tuple[str | None, set[str]]":
    """Diff this turn's discard pile against what we last knew, given what we
    ourselves discarded since then. Returns (taken_card_or_None, newly_discarded_by_opponent).

    ``taken_card_or_None`` is ``my_last_discard`` if it vanished from the pile
    (the opponent preferred it over a blind stock draw); the "expected" pile
    used for the second half folds ``my_last_discard`` in first so our own
    contribution is never misread as the opponent's.
    """
    taken = my_last_discard if (my_last_discard and my_last_discard not in pile) else None
    expected = known_pile | ({my_last_discard} if my_last_discard else set())
    opponent_discarded = pile - expected
    return taken, opponent_discarded


def render_opponent_note(taken_history: "list[str]", discarded_history: "list[str]") -> str:
    """Compact rolling summary of what the opponent's draws/discards reveal —
    see module docstring. Empty string if nothing's been observed yet."""
    lines = []
    if taken_history:
        ranks = ",".join(sorted({c[0].upper() for c in taken_history}))
        suits = ",".join(sorted({c[1].lower() for c in taken_history}))
        lines.append(
            f"Opponent took my discards: {' '.join(taken_history)} -- "
            f"likely building melds near rank(s) {ranks} or suit(s) {suits}; "
            f"avoid feeding more of those."
        )
    if discarded_history:
        lines.append(
            f"Opponent's own discards: {' '.join(discarded_history)} -- "
            f"probably doesn't need those ranks/suits, safer to discard similar cards."
        )
    return "\n".join(lines)


##############################################################################################
# Strategy
##############################################################################################

def _hand_stats(hand: list[str]) -> tuple[dict[str, int], set[str]]:
    rank_counts: dict[str, int] = Counter(get_rank(c) for c in hand)
    adj_cards = {c for c in hand if is_adjacent_in_suit(c, hand)}
    return rank_counts, adj_cards


def discard_score(card: str, hand: list[str], meld_cards: set[str],
                  rank_counts: dict[str, int], adj_cards: set[str],
                  dead_cards: Optional[set[str]] = None) -> int:
    score = card_value(card)
    if card in meld_cards:
        score -= 15
    has_pair = rank_counts[get_rank(card)] >= 2
    has_adj  = card in adj_cards
    if dead_cards is not None:
        if has_pair and partial_set_can_complete(card, hand, dead_cards):
            score -= 8
        if has_adj and partial_run_can_complete(card, hand, dead_cards):
            score -= 5
    else:
        if has_pair:
            score -= 8
        if has_adj:
            score -= 5
    return score


def choose_discard(hand: list[str], legal: list[tuple[str, str]], deadwood: int,
                   knock_card: int, dead_cards: Optional[set[str]] = None) -> str:
    if deadwood <= knock_card:
        knock_id = next((aid for aid, lbl in legal if lbl.strip().lower() == 'knock'), None)
        if knock_id:
            return knock_id

    meld_cards = get_optimal_meld_cards(hand)
    rank_counts, adj_cards = _hand_stats(hand)

    best_id, best_score = None, None
    for aid, label in legal:
        card_match = _RE_CARD_EXACT.match(label.strip())
        if not card_match:
            continue
        card = card_match.group(1)
        s = discard_score(card, hand, meld_cards, rank_counts, adj_cards, dead_cards)
        if best_score is None or s > best_score:
            best_score = s
            best_id = aid

    return best_id or legal[0][0]


def choose_draw(hand: list[str], upcard: str, legal: list[tuple[str, str]]) -> str:
    upcard_id = next((aid for aid, lbl in legal if 'Draw upcard' in lbl), None)
    stock_id  = next((aid for aid, lbl in legal if 'Draw stock'  in lbl), None)
    pass_id   = next((aid for aid, lbl in legal if lbl.strip() == 'Pass'), None)

    if upcard and upcard != 'XX' and upcard_id:
        if meld_potential(upcard, hand) > 0:
            return upcard_id

    if pass_id and not stock_id:
        return pass_id

    return stock_id or upcard_id or legal[0][0]


def choose_meld_or_layoff_action(legal: list[tuple[str, str]], hand: list[str],
                                  dead_cards: Optional[set[str]] = None) -> str:
    pass_id = next((aid for aid, lbl in legal if lbl.strip().lower() == 'pass'), None)

    for aid, label in legal:
        if _RE_MELD_GROUP.match(label.strip()):
            return aid

    meld_cards = get_optimal_meld_cards(hand)
    rank_counts, adj_cards = _hand_stats(hand)
    best_id, best_score = None, None
    for aid, label in legal:
        if aid == pass_id:
            continue
        card_match = _RE_CARD_EXACT.match(label.strip())
        if not card_match:
            continue
        card = card_match.group(1)
        s = discard_score(card, hand, meld_cards, rank_counts, adj_cards, dead_cards)
        if best_score is None or s > best_score:
            best_score = s
            best_id = aid

    return best_id or pass_id or legal[0][0]


##############################################################################################
# Expert action selector + episode runner
##############################################################################################

def _build_observation_envelope(state, player_id: int) -> str:
    """Render the same "Current State:\\n...You are Player N.\\nLegal Actions:\\n..."
    envelope the old mcts-api server returned, directly off a live pyspiel
    gin_rummy state.

    pyspiel's gin_rummy ``observation_string(player_id)`` is byte-identical to
    what the server returned as the raw observation body (confirmed by direct
    testing: same "Phase: ...", "Player{pid}: Deadwood=N" bordered hand grid,
    "Upcard:"/"Discard pile:"/"Knock card:" lines) — the server was a thin
    HTTP wrapper around this exact pyspiel call. Every parser below
    (``get_expert_action`` and its ``choose_*`` helpers, plus
    ``split_normalized_observation`` in pvp_format.py) operates on this single
    text blob via regex and is otherwise UNCHANGED from before; only the
    source of the blob changed, from HTTP+reformatting to direct pyspiel.
    """
    legal = state.legal_actions(player_id)
    legal_lines = "\n".join(f"{a} -> {state.action_to_string(player_id, a)}" for a in legal)
    return (
        f"Current State:\n{state.observation_string(player_id)}"
        f"You are Player {player_id}.\nLegal Actions:\n{legal_lines}\n"
    )


def get_expert_action(messages: list[dict]) -> str:
    obs        = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    phase      = parse_phase(obs)
    hand       = parse_hand(obs)
    upcard     = parse_upcard(obs)
    deadwood   = parse_deadwood(obs)
    knock_card = parse_knock_card(obs)
    legal      = parse_legal_actions(obs)
    dead_cards = parse_discard_pile(obs, upcard)

    if not legal:
        return "54"

    if phase in ("Draw", "FirstUpcard"):
        return choose_draw(hand, upcard, legal)

    if phase == "Discard":
        return choose_discard(hand, legal, deadwood, knock_card, dead_cards)

    if phase in ("Knock", "Layoff", "Wall"):
        return choose_meld_or_layoff_action(legal, hand, dead_cards)

    return legal[0][0]


def _build_tool_example(
    observation: str, action_id: int, system_prompt: str = _SYSTEM_PROMPT, memory_note: "str | None" = None,
) -> dict:
    """Build one stateless {system, user, assistant(tool_calls)} training example.

    ``system_prompt`` defaults to the always-empty-memory block; pass one from
    ``build_full_system_prompt(_GAME_NAME, memories=...)`` to render the
    current opponent-tracking note (see ``generate_expert_episode``). When
    ``memory_note`` is given, the assistant's response writes it to the
    working-memory slot in the SAME tool_calls list as game_action, mirroring
    core/pvp/bot.py's "one response, optionally edit memory, then commit a
    move" turn shape.
    """
    state_desc, player_id, legal_actions = split_normalized_observation(observation)
    tools = build_pvp_tools([aid for aid, _ in legal_actions])
    tool_calls = []
    if memory_note is not None:
        tool_calls.append({
            "type": "function",
            "function": {
                "name": _WORKING_REWRITE_TOOL,
                "arguments": json.dumps({"slot": _WORKING_NOTE_SLOT, "content": memory_note}),
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


def generate_expert_episode(
    game_id: int,
    max_turn: int = 200,
) -> list[dict]:
    """
    Run one Gin Rummy game on a real pyspiel state using the expert policy
    against an in-process MCTS opponent.

    Returns a list of independent per-turn training examples (one per action
    taken by the expert), each shaped
    ``{"messages": [system, user, assistant(tool_calls=[...,game_action])], "tools": [...]}``.
    The system prompt's memory block carries the running opponent-tracking
    note (see module docstring) once anything's been observed. Returns ``[]``
    on failure (e.g. a mid-game error).

    ``max_turn`` bounds the number of TEACHER decisions (one per phase:
    Draw/Discard/Knock/Layoff/Wall each count separately, exactly as the old
    HTTP-server loop counted one /step call per teacher decision) — not raw
    plies. The opponent's full turn (however many of its own phase-decisions
    it takes) runs to completion between two teacher decisions and is not
    counted against the budget, matching old semantics.
    """
    rng = random.Random(game_id)
    teacher_seat = rng.choice((0, 1))
    mcts_simulations = rng.randint(_MCTS_SIMS_MIN, _MCTS_SIMS_MAX)

    game = _AGENT.load_game(_AGENT.generate_params(config_id_for_task_id(game_id)))
    state = game.new_initial_state()
    opponent_bot = make_mcts_bot(game, mcts_simulations, seed=game_id)

    examples: list[dict] = []
    memories = default_memories()

    # Opponent-tracking state (see module docstring / pile_signal). known_pile
    # starts None: the very first observation has nothing to diff against, so
    # it's just recorded, not treated as an opponent action.
    known_pile: "set[str] | None" = None
    my_last_discard: "str | None" = None
    taken_history: list[str] = []
    discarded_history: list[str] = []

    def _resolve_until_teacher_turn() -> bool:
        """Advance past any chance nodes / opponent decisions. Returns False
        if the game ended (naturally, or because the opponent's MCTS search
        failed -- see mcts_step_or_none; in that case we stop the episode
        here rather than substitute a fabricated opponent move and keep
        going) before reaching another teacher decision."""
        while not state.is_terminal() and (state.is_chance_node() or state.current_player() != teacher_seat):
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(outcomes, weights=probs)[0])
            else:
                action = mcts_step_or_none(opponent_bot, state)
                if action is None:
                    _log(f"[gin_rummy_trajectories] Opponent MCTS search failed (game {game_id}), "
                         "truncating episode here")
                    return False
                state.apply_action(action)
        return not state.is_terminal()

    try:
        if not _resolve_until_teacher_turn():
            return []

        for _ in range(max_turn):
            observation = _build_observation_envelope(state, teacher_seat)

            upcard = parse_upcard(observation)
            pile = parse_discard_pile(observation, upcard)
            memory_note = None
            if known_pile is None:
                known_pile = set(pile)
            else:
                taken, opponent_discarded = pile_signal(known_pile, my_last_discard, pile)
                changed = bool(taken or opponent_discarded)
                if taken:
                    taken_history = (taken_history + [taken])[-_NOTE_HISTORY_LEN:]
                for c in opponent_discarded:
                    discarded_history = (discarded_history + [c])[-_NOTE_HISTORY_LEN:]
                known_pile = set(pile)
                if changed:
                    memory_note = render_opponent_note(taken_history, discarded_history)
                    memories[MemoryArea.WORKING].rewrite(_WORKING_NOTE_SLOT, memory_note)
            my_last_discard = None

            system_prompt = (
                build_full_system_prompt(_GAME_NAME, memories=memories)
                if (taken_history or discarded_history) else _SYSTEM_PROMPT
            )

            action = get_expert_action([{"role": "user", "content": observation}])

            try:
                action_id = int(action)
                examples.append(_build_tool_example(observation, action_id, system_prompt, memory_note))
            except Exception as exc:
                _log(f"[gin_rummy_trajectories] Failed to build example (game {game_id}): {exc}")

            if parse_phase(observation) == "Discard":
                label = next((lbl for aid, lbl in parse_legal_actions(observation) if aid == action), None)
                if label and _RE_CARD_EXACT.match(label.strip()):
                    my_last_discard = label.strip()

            state.apply_action(action_id)

            if not _resolve_until_teacher_turn():
                break
        else:
            _log(f"[gin_rummy_trajectories] max_turn={max_turn} reached (game {game_id})")
    except Exception as exc:
        _log(f"[gin_rummy_trajectories] Failed to build episode (game {game_id}): {exc}")
        return []

    return examples
