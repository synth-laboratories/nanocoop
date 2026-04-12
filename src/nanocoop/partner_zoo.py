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
        if not observation.shared_pot_known and observation.private_pot is not None:
            return "SHARE_POT"
        if not observation.shared_recipe_known and observation.private_recipe is not None:
            return "SHARE_RECIPE"
        if observation.ingredient_ready and observation.pot_ready and observation.dish_ready and not observation.plated:
            return "PLATE_SOUP"
        if observation.plated and not observation.delivered:
            return "SERVE_SOUP"
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
        if not observation.shared_pot_known and observation.private_pot is not None:
            return "SHARE_POT"
        if not observation.pot_ready:
            return "PREP_POT"
        if observation.ingredient_ready and observation.pot_ready and observation.dish_ready and not observation.plated:
            return "PLATE_SOUP"
        if observation.plated and not observation.delivered:
            return "SERVE_SOUP"
        if not observation.shared_recipe_known and observation.private_recipe is not None:
            return "SHARE_RECIPE"
        if not observation.dish_ready:
            return "FETCH_DISH"
        return "WAIT"


@dataclass
class HandoffPartner(BasePartner):
    name: str = "handoff"

    def act(self, observation: Observation) -> str:
        if not observation.shared_recipe_known and observation.private_recipe is not None:
            return "SHARE_RECIPE"
        if not observation.shared_pot_known and observation.private_pot is not None:
            return "SHARE_POT"
        if observation.ingredient_ready and observation.pot_ready and observation.dish_ready and not observation.plated:
            return "PLATE_SOUP"
        if observation.plated and not observation.delivered:
            return "SERVE_SOUP"
        if observation.last_partner_action == "FETCH_DISH" and not observation.ingredient_ready:
            return "FETCH_INGREDIENT"
        if not observation.ingredient_ready:
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
        if not observation.shared_recipe_known and observation.private_recipe is not None:
            return "SHARE_RECIPE"
        if not observation.shared_pot_known and observation.private_pot is not None:
            return "SHARE_POT"
        if not observation.ingredient_ready:
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
