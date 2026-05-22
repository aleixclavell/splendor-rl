
from typing import override

from splendor.splendor.types import ActionType
from splendor.template import Agent
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.gym.envs.actions import Action

class H1Agent(Agent):
    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:     
        # Prioritat 1: comprar (max punts; en empat, cost mínim)
        buy_actions = [a for a in actions if a["type"] in ("buy_available", "buy_reserve")]
        if buy_actions:
            return max(
                buy_actions,
                key=lambda a: (a["card"].points, -sum(a["card"].cost.values()))
            )

        # Prioritat 2: agafar gemmes (màxim de gemmes)
        collect_actions = [a for a in actions if a["type"].startswith("collect")]
        if collect_actions:
            return max(collect_actions, key=lambda a: sum(a["collected_gems"].values()))

        # Prioritat 3: reservar (només si la carta val ≥3 punts)
        reserve_actions = [a for a in actions if a["type"] == "reserve"]
        if reserve_actions:
            best_reserve = max(reserve_actions, key=lambda a: a["card"].points)
            if best_reserve["card"].points >= 3:
                return best_reserve

        return actions[0]      
    
myAgent = H1Agent  # pylint: disable=invalid-name