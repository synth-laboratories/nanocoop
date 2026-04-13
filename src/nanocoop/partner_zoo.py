from __future__ import annotations

import random
from dataclasses import dataclass

from nanocoop.schema import Observation


@dataclass
class BasePartner:
    name: str

    def act(self, observation: Observation) -> str:
        raise NotImplementedError


@dataclass
class CourierPartner(BasePartner):
    name: str = "courier"

    def act(self, observation: Observation) -> str:
        carrying = str(observation.metadata.get("inventory", "empty"))
        if "cooked_soup" in carrying and "plate" in carrying:
            return "SERVE_SOUP"
        if "ingredient" in carrying:
            return "PREP_POT"
        if "plate" in carrying and observation.pot_ready:
            return "PLATE_SOUP"
        can_reach_ingredient = bool(observation.metadata.get("can_reach_ingredient", True))
        pot_count = int(observation.metadata.get("pot_ingredient_count", 0))
        pot_full = bool(observation.metadata.get("pot_full", pot_count >= 3))
        if observation.ingredient_ready and observation.pot_ready and observation.dish_ready and not observation.plated:
            return "PLATE_SOUP"
        if observation.plated and not observation.delivered:
            return "SERVE_SOUP"
        if not pot_full and can_reach_ingredient:
            return "FETCH_INGREDIENT"
        if not observation.dish_ready:
            return "FETCH_DISH"
        if not observation.pot_ready:
            return "PREP_POT"
        if not observation.ingredient_ready:
            return "FETCH_INGREDIENT"
        return "WAIT"


@dataclass
class PotterPartner(BasePartner):
    name: str = "potter"

    def act(self, observation: Observation) -> str:
        carrying = str(observation.metadata.get("inventory", "empty"))
        if "cooked_soup" in carrying and "plate" in carrying:
            return "SERVE_SOUP"
        if "ingredient" in carrying:
            return "PREP_POT"
        if "plate" in carrying and observation.pot_ready:
            return "PLATE_SOUP"
        can_reach_ingredient = bool(observation.metadata.get("can_reach_ingredient", True))
        pot_count = int(observation.metadata.get("pot_ingredient_count", 0))
        pot_full = bool(observation.metadata.get("pot_full", pot_count >= 3))
        if not pot_full and can_reach_ingredient:
            return "FETCH_INGREDIENT"
        if not observation.pot_ready:
            return "PREP_POT"
        if observation.ingredient_ready and observation.pot_ready and observation.dish_ready and not observation.plated:
            return "PLATE_SOUP"
        if observation.plated and not observation.delivered:
            return "SERVE_SOUP"
        if not observation.dish_ready:
            return "FETCH_DISH"
        return "WAIT"


@dataclass
class HandoffPartner(BasePartner):
    name: str = "handoff"

    def act(self, observation: Observation) -> str:
        carrying = str(observation.metadata.get("inventory", "empty"))
        if "cooked_soup" in carrying and "plate" in carrying:
            return "SERVE_SOUP"
        if "ingredient" in carrying:
            return "PREP_POT"
        if "plate" in carrying and observation.pot_ready:
            return "PLATE_SOUP"
        can_reach_ingredient = bool(observation.metadata.get("can_reach_ingredient", True))
        pot_count = int(observation.metadata.get("pot_ingredient_count", 0))
        pot_full = bool(observation.metadata.get("pot_full", pot_count >= 3))
        if observation.ingredient_ready and observation.pot_ready and observation.dish_ready and not observation.plated:
            return "PLATE_SOUP"
        if observation.plated and not observation.delivered:
            return "SERVE_SOUP"
        if observation.last_partner_action == "FETCH_DISH" and not pot_full:
            if can_reach_ingredient:
                return "FETCH_INGREDIENT"
            return "PREP_POT"
        if not pot_full and can_reach_ingredient:
            return "FETCH_INGREDIENT"
        if not observation.pot_ready:
            return "PREP_POT"
        if not observation.dish_ready:
            return "FETCH_DISH"
        return "WAIT"


@dataclass
class NoisyPartner(BasePartner):
    name: str = "noisy"
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def act(self, observation: Observation) -> str:
        if self._rng.random() < 0.18:
            return "WAIT"
        if self._rng.random() < 0.1 and observation.last_partner_action is not None:
            return observation.last_partner_action
        carrying = str(observation.metadata.get("inventory", "empty"))
        if "cooked_soup" in carrying and "plate" in carrying:
            return "SERVE_SOUP"
        if "ingredient" in carrying:
            return "PREP_POT"
        if "plate" in carrying and observation.pot_ready:
            return "PLATE_SOUP"
        can_reach_ingredient = bool(observation.metadata.get("can_reach_ingredient", True))
        pot_count = int(observation.metadata.get("pot_ingredient_count", 0))
        pot_full = bool(observation.metadata.get("pot_full", pot_count >= 3))
        if not pot_full and can_reach_ingredient:
            return "FETCH_INGREDIENT"
        if not observation.pot_ready:
            return "PREP_POT"
        if not observation.dish_ready:
            return "FETCH_DISH"
        if observation.ingredient_ready and observation.pot_ready and observation.dish_ready and not observation.plated:
            return "PLATE_SOUP"
        if observation.plated and not observation.delivered:
            return "SERVE_SOUP"
        return "WAIT"


def make_partner(name: str, seed: int = 0):
    if name == "courier":
        return CourierPartner()
    if name == "potter":
        return PotterPartner()
    if name == "handoff":
        return HandoffPartner()
    if name == "noisy":
        return NoisyPartner(seed=seed)
    raise ValueError(f"Unknown partner: {name}")
