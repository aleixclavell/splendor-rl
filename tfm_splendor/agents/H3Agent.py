from itertools import combinations
from typing import override

from splendor.splendor.types import ActionType
from splendor.template import Agent
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState


GEM_COLORS = ["black", "red", "green", "blue", "white"]
WIN_SCORE = 15


class H3Agent(Agent):
    """
    Estratègia:
    - Reservar cartes bones aviat.
    - Agafar 3 fitxes als primers torns.
    - Especialitzar-se en 2 colors.
    - Prioritzar nobles amb requisits solapats.
    - Comprar ràpid cartes amb punts.
    """

    # ------------------------------------------------------------------ #
    # Estat i càlculs bàsics
    # ------------------------------------------------------------------ #

    def _owned_cards_count(self, agent) -> int:
        return sum(len(agent.cards.get(c, [])) for c in GEM_COLORS)

    def _is_early_game(self, agent) -> bool:
        return self._owned_cards_count(agent) <= 4 and agent.score <= 3

    def _effective_gems(self, agent) -> dict:
        eff = dict(agent.gems)
        for color in GEM_COLORS:
            eff[color] = eff.get(color, 0) + len(agent.cards.get(color, []))
        return eff

    def _shortfall(self, card, agent) -> dict:
        eff = self._effective_gems(agent)
        yellow = eff.get("yellow", 0)

        short = {}
        for color in GEM_COLORS:
            short[color] = max(0, card.cost.get(color, 0) - eff.get(color, 0))

        for color in sorted(short, key=lambda c: -short[c]):
            used = min(yellow, short[color])
            short[color] -= used
            yellow -= used
            if yellow == 0:
                break

        return {c: v for c, v in short.items() if v > 0}

    def _total_shortfall(self, card, agent) -> int:
        return sum(self._shortfall(card, agent).values())

    def _turns_to_buy(self, card, agent) -> int:
        missing = self._total_shortfall(card, agent)
        if missing == 0:
            return 0
        return max(1, -(-missing // 2))

    def _all_board_cards(self, board):
        return [card for tier in board.dealt for card in tier if card is not None]

    def _reserved_cards(self, agent):
        return agent.cards.get("yellow", [])

    # ------------------------------------------------------------------ #
    # Nobles i colors objectiu
    # ------------------------------------------------------------------ #

    def _best_noble_pair(self, agent, board):
        nobles = board.nobles
        if not nobles:
            return []

        owned = {c: len(agent.cards.get(c, [])) for c in GEM_COLORS}

        def noble_progress(cost):
            return sum(min(owned[c], cost.get(c, 0)) for c in GEM_COLORS)

        def pair_score(pair):
            (_, c1), (_, c2) = pair
            overlap = sum(min(c1.get(c, 0), c2.get(c, 0)) for c in GEM_COLORS)
            progress = noble_progress(c1) + noble_progress(c2)
            total_req = sum(c1.values()) + sum(c2.values())
            return overlap * 3 + progress * 2 - total_req * 0.15

        if len(nobles) == 1:
            return nobles

        return list(max(combinations(nobles, 2), key=pair_score))

    def _target_colors(self, agent, board) -> list[str]:
        noble_pair = self._best_noble_pair(agent, board)

        demand = {c: 0.0 for c in GEM_COLORS}

        for _, cost in noble_pair:
            for color in GEM_COLORS:
                demand[color] += cost.get(color, 0)

        # Bonus pels colors que ja tenim: reforça l’especialització.
        for color in GEM_COLORS:
            demand[color] += len(agent.cards.get(color, [])) * 1.25

        # Bonus per colors comuns entre nobles.
        if len(noble_pair) == 2:
            c1 = noble_pair[0][1]
            c2 = noble_pair[1][1]
            for color in GEM_COLORS:
                if c1.get(color, 0) > 0 and c2.get(color, 0) > 0:
                    demand[color] += 2.0

        return sorted(GEM_COLORS, key=lambda c: -demand[c])[:2]

    def _noble_gain(self, card, agent, board) -> float:
        noble_pair = self._best_noble_pair(agent, board)
        if not noble_pair:
            return 0.0

        owned = {c: len(agent.cards.get(c, [])) for c in GEM_COLORS}
        color = card.colour

        gain = 0.0
        for _, cost in noble_pair:
            if color not in cost:
                continue

            before = sum(min(owned[c], cost.get(c, 0)) for c in GEM_COLORS)
            owned[color] += 1
            after = sum(min(owned[c], cost.get(c, 0)) for c in GEM_COLORS)
            owned[color] -= 1

            gain += after - before

            # Si la carta deixa el noble molt a prop, puja molt el valor.
            missing_after = sum(
                max(0, cost.get(c, 0) - (owned[c] + (1 if c == color else 0)))
                for c in GEM_COLORS
            )
            if missing_after <= 1:
                gain += 2.5

        return gain

    # ------------------------------------------------------------------ #
    # Valoració de cartes
    # ------------------------------------------------------------------ #

    def _single_color_cost_bonus(self, card) -> float:
        non_zero = [(c, v) for c, v in card.cost.items() if c in GEM_COLORS and v > 0]
        if len(non_zero) != 1:
            return 0.0

        _, amount = non_zero[0]
        if amount >= 4:
            return 3.0
        if amount == 3:
            return 1.5
        return 0.5

    def _card_score(self, card, agent, board) -> float:
        target_colors = self._target_colors(agent, board)
        turns = self._turns_to_buy(card, agent)

        vp_score = card.points * 3.0
        noble_score = self._noble_gain(card, agent, board) * 2.0
        color_score = 1.5 if card.colour in target_colors else 0.0
        tier_score = card.deck_id * 0.35
        same_color_cost = self._single_color_cost_bonus(card)

        # Si ajuda a tancar la partida, pesa més.
        win_pressure = 0.0
        points_needed = WIN_SCORE - agent.score
        if card.points >= points_needed:
            win_pressure = 10.0
        elif agent.score >= 10 and card.points >= 3:
            win_pressure = 3.0

        return (
            vp_score
            + noble_score
            + color_score
            + tier_score
            + same_color_cost
            + win_pressure
        ) / (turns + 1)

    def _best_target(self, agent, board):
        candidates = self._all_board_cards(board) + self._reserved_cards(agent)
        if not candidates:
            return None
        return max(candidates, key=lambda c: self._card_score(c, agent, board))

    def _best_reserve_action(self, reserve_actions, agent, board):
        if not reserve_actions:
            return None

        target_colors = self._target_colors(agent, board)

        def score(action):
            card = action["card"]

            value = self._card_score(card, agent, board)

            # Reservar cartes cares d’un sol color és part explícita de l’estratègia.
            value += self._single_color_cost_bonus(card) * 2.0

            # Preferim reservar cartes que donin punts.
            value += card.points * 1.5

            # Preferim colors de la nostra especialització.
            if card.colour in target_colors:
                value += 2.0

            # Penalitza cartes massa llunyanes sense punts.
            value -= self._turns_to_buy(card, agent) * 0.4

            return value

        return max(reserve_actions, key=score)

    # ------------------------------------------------------------------ #
    # Accions concretes
    # ------------------------------------------------------------------ #

    def _winning_action(self, buy_actions, agent):
        for action in buy_actions:
            noble_points = 3 if action.get("noble") else 0
            if agent.score + action["card"].points + noble_points >= WIN_SCORE:
                return action
        return None

    def _best_buy_action(self, buy_actions, agent, board):
        return max(
            buy_actions,
            key=lambda a: (
                a["card"].points + (3 if a.get("noble") else 0),
                self._card_score(a["card"], agent, board),
                -self._total_shortfall(a["card"], agent),
            ),
        )

    def _directed_collect(self, target, agent, board, collect_actions):
        if not collect_actions:
            return None

        target_colors = self._target_colors(agent, board)
        short = self._shortfall(target, agent) if target else {}

        def score(action):
            collected = action.get("collected_gems", {})
            returned = action.get("returned_gems", {})

            total_taken = sum(collected.values())
            distinct_taken = sum(1 for v in collected.values() if v > 0)

            value = 0.0

            # Principi de partida: agafar 3 fitxes és molt bo.
            if self._is_early_game(agent) and action["type"] == "collect_diff":
                if total_taken == 3 and distinct_taken == 3:
                    value += 5.0

            # Fitxes que falten per comprar l’objectiu.
            for color, amount in collected.items():
                if color in short:
                    value += amount * 3.0
                if color in target_colors:
                    value += amount * 1.5

            # Bonus especial: si només falta un color, collect_same és molt valuós.
            if action["type"] == "collect_same" and len(short) == 1:
                only_color = next(iter(short))
                useful = min(collected.get(only_color, 0), short[only_color])
                value += useful * 2.0

            # Penalitza retornar colors importants.
            for color, amount in returned.items():
                if color in short:
                    value -= amount * 4.0
                if color in target_colors:
                    value -= amount * 2.0

            return value

        best = max(collect_actions, key=score)
        return best if score(best) > 0 else None

    def _opponent_close_to_win(self, game_state) -> bool:
        return any(
            other.score >= WIN_SCORE - 3
            for other in game_state.agents
            if other.id != self.id
        )

    # ------------------------------------------------------------------ #
    # Decisió principal
    # ------------------------------------------------------------------ #

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        agent = game_state.agents[self.id]
        board = game_state.board

        buy_actions = [
            a for a in actions
            if a["type"] in ("buy_available", "buy_reserve")
        ]
        collect_actions = [
            a for a in actions
            if a["type"].startswith("collect")
        ]
        reserve_actions = [
            a for a in actions
            if a["type"] == "reserve"
        ]

        reserved_count = len(agent.cards.get("yellow", []))
        early = self._is_early_game(agent)
        target = self._best_target(agent, board)

        # 1. Guanyar immediatament.
        if buy_actions:
            win = self._winning_action(buy_actions, agent)
            if win:
                return win

        # 2. Comprar si hi ha una bona carta disponible.
        if buy_actions:
            if target:
                target_buy = next(
                    (
                        a for a in buy_actions
                        if a["card"].code == target.code
                    ),
                    None,
                )
                if target_buy:
                    return target_buy

            return self._best_buy_action(buy_actions, agent, board)

        # 3. Reservar aviat cartes clau.
        if reserve_actions and reserved_count < 3:
            best_reserve = self._best_reserve_action(reserve_actions, agent, board)

            if best_reserve:
                card = best_reserve["card"]
                reserve_value = self._card_score(card, agent, board)

                should_reserve = (
                    early
                    or card.points >= 3
                    or self._single_color_cost_bonus(card) >= 3
                    or self._opponent_close_to_win(game_state)
                )

                if should_reserve and reserve_value > 1.2:
                    return best_reserve

        # 4. Agafar fitxes dirigides a l’objectiu.
        if collect_actions:
            collect = self._directed_collect(target, agent, board, collect_actions)
            if collect:
                return collect

        # 5. Si encara podem reservar una carta valuosa, fem-ho.
        if reserve_actions and reserved_count < 3:
            best_reserve = self._best_reserve_action(reserve_actions, agent, board)
            if best_reserve and best_reserve["card"].points >= 3:
                return best_reserve

        # 6. Fallback: agafar el màxim nombre de fitxes.
        if collect_actions:
            return max(
                collect_actions,
                key=lambda a: sum(a.get("collected_gems", {}).values()),
            )

        return actions[0]


myAgent = H3Agent  # pylint: disable=invalid-name