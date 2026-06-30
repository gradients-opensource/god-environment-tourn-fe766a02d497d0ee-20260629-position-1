"""Game mechanics — ported from /ephemeral/G.O.D core/pvp/{agents,game_eval,scoring}.py
and core/pvp/baseline.py, trimmed to what SFT trajectory generation needs (no
LLMBot, no chat/tool-call parsing, no SGLang). See docs/PYSPIEL_NATIVE_DATAGEN_PLAN.md.

This is the single source of truth for: which pyspiel game variant a given
game id plays, how to load it, how to render its state, and how to score its
terminal outcome — replacing the per-env HTTP env-server + regex-parsing
plumbing the trajectory generators used previously. Decision POLICY (which
move to pick) is NOT here — it stays in each game's *_trajectories.py /
*_policy.py / *_minimax.py, untouched; this module only gets a generator from
"a game id" to "a live pyspiel.State it can read directly".
"""

import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from pydantic import BaseModel

from envs.pvp_constants import PVP_CONFIG_ID_DIVISOR

# ---------------------------------------------------------------------------
# Game-variant params — verbatim port of core/models/pvp_models.py's GameParams
# subclasses (trimmed: no `game` discriminator tag, not needed outside the
# tagged-union round-trip that exists for the eval-side pydantic models).
# ---------------------------------------------------------------------------


class GameParams(BaseModel):
    """Base for a game's pyspiel.load_game() parameters."""

    def to_pyspiel(self) -> dict[str, int | str | bool]:
        return self.model_dump()


class LiarsDiceParams(GameParams):
    players: int = 2
    numdice: int = 5


class LeducPokerParams(GameParams):
    players: int = 2


class GinRummyParams(GameParams):
    hand_size: int
    knock_card: int


class OthelloParams(GameParams):
    pass


class GoofspielParams(GameParams):
    players: int = 2
    num_cards: int
    imp_info: bool = True
    points_order: Literal["random", "ascending", "descending"] = "random"
    returns_type: Literal["win_loss", "total_points", "point_difference"] = "win_loss"


class ClobberParams(GameParams):
    rows: int
    columns: int


# ---------------------------------------------------------------------------
# Per-game agents — verbatim port of core/pvp/agents.py, minus the rules-text/
# system-prompt methods (pvp_format.py already owns that, byte-for-byte from
# the same pvp_assets/pvp_game_prompts.yml). format_state/load_game/
# generate_params/setup_initial_state are kept identical to eval.
# ---------------------------------------------------------------------------


class BaseGameAgent(ABC):
    """Abstract base for game-specific pyspiel wiring."""

    @property
    @abstractmethod
    def game_name(self) -> str:
        ...

    @abstractmethod
    def generate_params(self, config_id: int) -> GameParams:
        """Generate pyspiel game parameters from a config variant id."""
        ...

    def setup_initial_state(self, state: pyspiel.State, seed: int) -> None:
        """Advance the fresh state before play starts. Default: no-op.

        Games with chance nodes (dice, card deals) get their per-game variety
        for free from the seed used to drive chance outcomes. Deterministic
        games with no chance nodes (e.g. othello) override this to inject
        seeded variety so the same seed reproduces the same start while
        different seeds diverge.
        """
        return None

    def load_game(self, params: GameParams) -> pyspiel.Game:
        """Build the pyspiel game this agent plays.

        Default: load game_name with params directly. Games whose native
        dynamics are simultaneous (goofspiel) override this to wrap the game
        so a sequential turn-by-turn loop can drive it.
        """
        return pyspiel.load_game(self.game_name, params.to_pyspiel())

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        """Format game state as text. Override for game-specific formatting."""
        try:
            return state.observation_string(player_id)
        except (RuntimeError, AttributeError):
            pass
        try:
            return state.information_state_string(player_id)
        except (RuntimeError, AttributeError):
            raise ValueError(
                f"Game {self.game_name} supports neither observation_string nor "
                f"information_state_string — override format_state() for this game"
            )


class LiarsDiceAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "liars_dice"

    def generate_params(self, config_id: int) -> GameParams:
        return LiarsDiceParams(players=2, numdice=5)

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        try:
            info_str = state.information_state_string(player_id)
        except (RuntimeError, AttributeError):
            return str(state)

        if not info_str:
            return str(state)

        parts = info_str.split()
        dice_part = parts[0]
        bid_parts = [p for p in parts[1:] if "-" in p]

        dice = [int(d) for d in dice_part if d.isdigit()]
        num_dice = len(dice)
        total_dice = num_dice * state.num_players()

        lines = [
            f"Your dice: {dice} (showing: {', '.join(map(str, dice))})",
            f"Dice per player: {num_dice}",
            f"Total dice in game: {total_dice}",
            f"Players: {state.num_players()}",
            f"Current player: Player {state.current_player()}",
        ]

        if bid_parts:
            last_bid = bid_parts[-1]
            quantity, face = last_bid.split("-")
            lines.append(
                f'\nCurrent bid: "{quantity}-{face}" '
                f"(at least {quantity} dice showing {face} across all players)"
            )
            lines.append("You can: (1) Make a higher bid, or (2) Call 'Liar'")
        else:
            lines.append("No bid yet - you must make the first bid")

        return "\n".join(lines)


class LeducPokerAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "leduc_poker"

    def generate_params(self, config_id: int) -> GameParams:
        return LeducPokerParams(players=2)

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        try:
            info_str = state.information_state_string(player_id)
        except (RuntimeError, AttributeError):
            return str(state)

        import re

        def _extract(pattern: str) -> str:
            match = re.search(pattern, info_str)
            return match.group(1) if match else ""

        def _card_name(card_id: int) -> str:
            ranks = ["J", "Q", "K", "A"]
            suits = ["♠", "♥"]
            rank_idx = card_id // 2
            suit_idx = card_id % 2
            if rank_idx < len(ranks):
                return f"{ranks[rank_idx]}{suits[suit_idx]}"
            return f"Card_{card_id}"

        private_card = _extract(r"\[Private: (-?\d+)\]")
        round_num = _extract(r"\[Round (\d+)\]")
        pot = _extract(r"\[Pot: (\d+)\]")
        money = _extract(r"\[Money: ([\d ]+)\]")
        public_card = _extract(r"\[Public: (-?\d+)\]")

        lines: list[str] = []

        if private_card and private_card != "-10000":
            lines.append(f"Your card: {_card_name(int(private_card))}")
        else:
            lines.append("Your card: (not dealt yet)")

        if public_card and public_card != "-10000":
            lines.append(f"Public card: {_card_name(int(public_card))}")
            if private_card and private_card != "-10000":
                if int(private_card) // 2 == int(public_card) // 2:
                    lines.append("Hand: PAIR")

        lines.append(f"Round: {round_num}/2")
        lines.append(f"Pot: {pot} chips")

        if money:
            chips = money.split()
            if len(chips) >= 2:
                lines.append(f"Your chips: {chips[player_id]}")
                lines.append(f"Opponent chips: {chips[1 - player_id]}")

        return "\n".join(lines)


class GinRummyAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "gin_rummy"

    def generate_params(self, config_id: int) -> GameParams:
        hand_var = (config_id // 3) % 3
        knock_var = config_id % 3
        return GinRummyParams(hand_size=7 + hand_var, knock_card=10 - knock_var)

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        return state.observation_string(player_id)


# Number of seeded random opening plies applied to an othello game, sampled
# from this inclusive range. Enough to diverge the opening tree for variety,
# few enough that positions stay balanced and game-like.
_OTHELLO_OPENING_PLIES = (2, 6)


class OthelloAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "othello"

    def generate_params(self, config_id: int) -> GameParams:
        return OthelloParams()

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        """Prefix the board with the player's colour.

        The observation only says whose turn it is ("Black (x) to play"), so
        without this line the model must infer its own colour — small models
        get it wrong and play for the opponent.
        """
        colour = "x (Black)" if player_id == 0 else "o (White)"
        return f"You play {colour}.\n{state.observation_string(player_id)}"

    def setup_initial_state(self, state: pyspiel.State, seed: int) -> None:
        """Apply a seeded number of uniformly-random legal opening moves.

        Othello is deterministic with no chance nodes, so every game would
        otherwise start from the identical board. Deriving the opening plies
        from the instance seed keeps games reproducible (same seed -> same
        start) while giving each seed a distinct mid-game position to play
        from.
        """
        rng = random.Random(seed)
        num_plies = rng.randint(*_OTHELLO_OPENING_PLIES)
        for _ in range(num_plies):
            if state.is_terminal():
                break
            legal_actions = state.legal_actions()
            if not legal_actions:
                break
            state.apply_action(rng.choice(legal_actions))


# Board sizes clobber is played on, selected per game from the config id so
# each game varies board size for SFT/eval diversity. Mirrors
# core/pvp/agents.py's _CLOBBER_BOARD_SIZES.
_CLOBBER_BOARD_SIZES = ((4, 5), (5, 5), (5, 6))

# Same opening-ply range Othello uses (above) -- Clobber is also deterministic
# with no chance nodes, so it needs the same seeded-random-opening treatment.
_CLOBBER_OPENING_PLIES = (2, 6)


class ClobberAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "clobber"

    def generate_params(self, config_id: int) -> GameParams:
        rows, columns = _CLOBBER_BOARD_SIZES[config_id % len(_CLOBBER_BOARD_SIZES)]
        return ClobberParams(rows=rows, columns=columns)

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        colour = "o (White)" if player_id == 0 else "x (Black)"
        return f"You play {colour}.\n{state.observation_string(player_id)}"

    def setup_initial_state(self, state: pyspiel.State, seed: int) -> None:
        """Apply a seeded number of uniformly-random legal opening moves.

        Same rationale as OthelloAgent.setup_initial_state above: clobber is
        deterministic with no chance nodes, so every game would otherwise
        start from the identical board.
        """
        rng = random.Random(seed)
        num_plies = rng.randint(*_CLOBBER_OPENING_PLIES)
        for _ in range(num_plies):
            if state.is_terminal():
                break
            legal_actions = state.legal_actions()
            if not legal_actions:
                break
            state.apply_action(rng.choice(legal_actions))


# Deck sizes goofspiel is played with, selected per game from the config id so
# each game varies board size (and thus length) for SFT/eval diversity. 5 is a
# short sharp game; 13 is the full standard deck.
_GOOFSPIEL_NUM_CARDS = (5, 8, 10, 13)


class GoofspielAgent(BaseGameAgent):
    """Goofspiel (a.k.a. the Game of Pure Strategy).

    OpenSpiel's goofspiel is a SIMULTANEOUS-move game; load_game wraps it via
    convert_to_turn_based so a sequential turn-by-turn loop can drive it,
    hiding each player's concurrent bid from the other (so simultaneity and
    fairness are preserved). Played with imp_info=True (opponent hand hidden)
    and returns_type=win_loss so terminal returns are zero-sum {-1, 0, 1},
    mapping straight to win/loss/draw.
    """

    @property
    def game_name(self) -> str:
        return "goofspiel"

    def generate_params(self, config_id: int) -> GameParams:
        num_cards = _GOOFSPIEL_NUM_CARDS[config_id % len(_GOOFSPIEL_NUM_CARDS)]
        return GoofspielParams(
            players=2,
            num_cards=num_cards,
            imp_info=True,
            points_order="random",
            returns_type="win_loss",
        )

    def load_game(self, params: GameParams) -> pyspiel.Game:
        """Load goofspiel and wrap its simultaneous moves into sequential turns."""
        return pyspiel.convert_to_turn_based(pyspiel.load_game(self.game_name, params.to_pyspiel()))

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        """Render the player's own view; imp_info keeps the opponent's hand hidden."""
        return f"You are Player {player_id} (P{player_id}).\n{state.observation_string(player_id)}"


AGENT_REGISTRY: dict[str, type[BaseGameAgent]] = {
    "liars_dice": LiarsDiceAgent,
    "leduc_poker": LeducPokerAgent,
    "gin_rummy": GinRummyAgent,
    "othello": OthelloAgent,
    "goofspiel": GoofspielAgent,
    "clobber": ClobberAgent,
}


def get_agent(game_name: str) -> BaseGameAgent:
    return AGENT_REGISTRY[game_name]()


# ---------------------------------------------------------------------------
# Game/config sampling — port of core/pvp/game_eval.py::config_id_for_seed.
#
# Eval derives a task_id from a seed via its own RNG, then takes
# `task_id % PVP_CONFIG_ID_DIVISOR` as the config id. This repo's generators
# already sample a real task_id directly from the env's task_id range (see
# shared_env.GAMES_TO_TASK_ID_RANGE) instead of re-deriving one from a seed —
# that task_id plays the exact same role eval's locally-derived one does, so
# we skip the redundant seed->task_id hop and apply the final mod directly.
# ---------------------------------------------------------------------------


def config_id_for_task_id(task_id: int) -> int:
    """task_id -> game-variant config id. Mirrors eval's final mapping step."""
    return task_id % PVP_CONFIG_ID_DIVISOR


# ---------------------------------------------------------------------------
# Outcome scoring — verbatim port of core/pvp/scoring.py::determine_outcome,
# returning the [0, 1] win=1/draw=0.5/loss=0 scale generate_trajectories.py
# already expects from generators for --wins-only/--sample-by-score.
# ---------------------------------------------------------------------------


class GameOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


_OUTCOME_SCORE = {GameOutcome.WIN: 1.0, GameOutcome.DRAW: 0.5, GameOutcome.LOSS: 0.0}


def determine_outcome(
    returns: list[float], player_id: int, is_zero_sum: bool, min_utility: float, max_utility: float
) -> GameOutcome:
    """Determine win/loss/draw for player_id from terminal returns.

    For zero-sum games, normalizes the player's return to [0, 1] and uses 0.5
    as the draw threshold. For general-sum, compares raw returns.
    """
    player_return = returns[player_id]
    opponent_return = returns[1 - player_id]

    if is_zero_sum:
        if max_utility > min_utility:
            score = (player_return - min_utility) / (max_utility - min_utility)
        else:
            score = 0.5

        if score > 0.5:
            return GameOutcome.WIN
        elif score < 0.5:
            return GameOutcome.LOSS
        return GameOutcome.DRAW

    if player_return > opponent_return:
        return GameOutcome.WIN
    elif player_return < opponent_return:
        return GameOutcome.LOSS
    return GameOutcome.DRAW


def score_for_player(state: pyspiel.State, player_id: int) -> float:
    """Terminal state -> [0, 1] score for player_id (1.0 win, 0.5 draw, 0.0 loss)."""
    game = state.get_game()
    game_type = game.get_type()
    outcome = determine_outcome(
        returns=state.returns(),
        player_id=player_id,
        is_zero_sum=game_type.utility == pyspiel.GameType.Utility.ZERO_SUM,
        min_utility=game.min_utility(),
        max_utility=game.max_utility(),
    )
    return _OUTCOME_SCORE[outcome]


# ---------------------------------------------------------------------------
# In-process MCTS opponent — verbatim port of core/pvp/baseline.py::_make_mcts_bot.
# No server, no concurrency cap: a plain Python object per worker process.
# ---------------------------------------------------------------------------

MCTS_UCT_C = 2.0


def make_mcts_bot(game: pyspiel.Game, simulations: int, seed: int) -> "mcts.MCTSBot":
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=np.random.RandomState(seed))
    return mcts.MCTSBot(
        game,
        uct_c=MCTS_UCT_C,
        max_simulations=simulations,
        evaluator=evaluator,
        random_state=np.random.RandomState(seed),
    )


def mcts_step_or_none(bot: "mcts.MCTSBot", state: pyspiel.State) -> "int | None":
    """``bot.step(state)``, returning ``None`` instead of raising if the
    search itself fails.

    Known failure mode (observed live, e.g. on gin_rummy): OpenSpiel's
    ``RandomRolloutEvaluator`` clones the real state and plays it out
    randomly to estimate a value; for some games' pyspiel implementations
    that random rollout can wander into a state the engine considers
    non-terminal but with zero legal actions for the current player —
    ``mcts.py``'s ``np.random.choice(working_state.legal_actions())`` then
    raises ``ValueError: 'a' cannot be empty unless no samples are taken``.
    This is a bug in the rollout's *simulated* continuation inside the
    opponent's search, not in the real game state we're actually playing.

    Callers should treat ``None`` as "stop the episode here" (keep whatever
    teacher-turn examples were already built, same as hitting max_turn) —
    NOT substitute a random move and keep going: a real MCTS bot never
    actually chose that move, so continuing would train the teacher's later
    turns against an opponent action it would never have faced, silently
    mixing fabricated-opponent states into otherwise-real trajectories.
    Truncating loses only the turns after the failure, not the ones already
    collected, and never feeds the teacher a state that didn't really occur.
    """
    try:
        return bot.step(state)
    except Exception:
        return None
