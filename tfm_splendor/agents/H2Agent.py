
from typing import override

from splendor.splendor.types import ActionType
from splendor.template import Agent
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState

GEM_COLORS = ["black", "red", "green", "blue", "white"]
WIN_SCORE = 15


class H2Agent(Agent): 

    def _effective_gems(self, agent) -> dict:
        """Gems + card-bonus counts (what the agent can spend)."""
        eff = dict(agent.gems)
        for color in GEM_COLORS:
            eff[color] = eff.get(color, 0) + len(agent.cards.get(color, []))
        return eff

    def _shortfall(self, card, agent) -> dict:
        """Per-color gem shortfall to buy a card, after bonuses and yellow wildcards."""
        eff = self._effective_gems(agent)
        yellow = eff.get("yellow", 0)
        short = {}
        for color in GEM_COLORS:
            deficit = max(0, card.cost.get(color, 0) - eff.get(color, 0))
            short[color] = deficit
        # Apply yellow wildcards to the largest deficits first
        for color in sorted(short, key=lambda c: -short[c]):
            use = min(yellow, short[color])
            short[color] -= use
            yellow -= use
            if yellow == 0:
                break
        return {c: v for c, v in short.items() if v > 0}

    def _total_shortfall(self, card, agent) -> int:
        return sum(self._shortfall(card, agent).values())

    def _turns_to_buy(self, card, agent) -> int:
        """Rough estimate: each turn collects ~2 gems on average."""
        needed = self._total_shortfall(card, agent)
        if needed == 0:
            return 0
        # collect_same gives 2 of one color; collect_diff up to 3 different
        # conservatively assume 2 gems/turn toward the right colors
        return max(1, -(-needed // 2))  # ceil division

    def _noble_contribution(self, card, board_nobles) -> int:
        """Number of available nobles that require this card's color."""
        return sum(1 for _, cost in board_nobles if card.colour in cost)

    def _card_score(self, card, agent, board_nobles) -> float:
        """Higher score = better card to target. Rewards VP, noble progress, cheap cost."""
        vp = card.points
        noble = self._noble_contribution(card, board_nobles)
        turns = self._turns_to_buy(card, agent)
        # Tier bonus: higher-tier cards have more VP per gem invested
        tier = card.deck_id * 0.3
        return (vp * 2.5 + noble * 1.5 + tier) / (turns + 1)

    def _all_board_cards(self, board):
        return [c for tier in board.dealt for c in tier if c is not None]

    def _best_target(self, agent, board):
        """Best card to focus on: board + reserved cards."""
        candidates = self._all_board_cards(board) + agent.cards.get("yellow", [])
        if not candidates:
            return None
        return max(candidates, key=lambda c: self._card_score(c, agent, board.nobles))

    def _directed_collect(self, target, agent, actions):
        """Pick the collect action that brings us closest to buying target."""
        short = self._shortfall(target, agent)
        if not short:
            return None
        collect_actions = [a for a in actions if a["type"].startswith("collect")]
        if not collect_actions:
            return None

        def score(action):
            collected = action.get("collected_gems", {})
            returned = action.get("returned_gems", {})
            gain = sum(collected.get(c, 0) for c in short)
            loss = sum(returned.get(c, 0) for c in short)
            return gain - loss * 2

        best = max(collect_actions, key=score)
        return best if score(best) > 0 else None

    def _winning_action(self, buy_actions, agent):
        """Return the first buy action that would reach WIN_SCORE."""
        for a in buy_actions:
            noble_pts = 3 if a.get("noble") else 0
            if agent.score + a["card"].points + noble_pts >= WIN_SCORE:
                return a
        return None

    def _opponent_threatening(self, game_state) -> bool:
        return any(
            ag.score >= WIN_SCORE - 3
            for ag in game_state.agents
            if ag.id != self.id
        )

    # ------------------------------------------------------------------ #
    # Main decision                                                        #
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

        buy_actions = [a for a in actions if a["type"] in ("buy_available", "buy_reserve")]
        collect_actions = [a for a in actions if a["type"].startswith("collect")]
        reserve_actions = [a for a in actions if a["type"] == "reserve"]

        # 1. Win immediately if possible
        if buy_actions:
            win = self._winning_action(buy_actions, agent)
            if win:
                return win

        # 2. Find the card to focus on
        target = self._best_target(agent, board)

        # 3. Buy: prefer target card, then best available by score
        if buy_actions:
            if target:
                target_buy = next(
                    (a for a in buy_actions if a["card"].code == target.code), None
                )
                if target_buy:
                    return target_buy
            # Buy best buyable card considering noble visits
            return max(
                buy_actions,
                key=lambda a: (
                    a["card"].points + (3 if a.get("noble") else 0),
                    self._noble_contribution(a["card"], board.nobles),
                    -self._total_shortfall(a["card"], agent),
                ),
            )

        # 4. Collect gems directed at the target card
        if target and collect_actions:
            best_collect = self._directed_collect(target, agent, actions)
            if best_collect:
                return best_collect

        # 5. Reserve: target card is worth reserving if opponent threatens or it scores high
        reserved_count = len(agent.cards.get("yellow", []))
        if reserve_actions and target and reserved_count < 3:
            target_reserve = next(
                (a for a in reserve_actions if a["card"].code == target.code), None
            )
            if target_reserve:
                if self._opponent_threatening(game_state) or target.points >= 3:
                    return target_reserve

        # 6. Fallback: collect as many gems as possible
        if collect_actions:
            return max(collect_actions, key=lambda a: sum(a["collected_gems"].values()))

        # 7. Reserve any card with high VP
        if reserve_actions and reserved_count < 3:
            best = max(reserve_actions, key=lambda a: a["card"].points)
            if best["card"].points >= 3:
                return best

        return actions[0]


myAgent = H2Agent  # pylint: disable=invalid-name
