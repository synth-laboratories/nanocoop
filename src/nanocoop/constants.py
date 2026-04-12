from __future__ import annotations

COOP_ACTIONS = (
    "SHARE_RECIPE",
    "SHARE_POT",
    "FETCH_INGREDIENT",
    "PREP_POT",
    "FETCH_DISH",
    "PLATE_SOUP",
    "SERVE_SOUP",
    "WAIT",
)

TRAIN_LAYOUTS = (
    "grounded_coord_simple",
    "grounded_coord_ring",
    "demo_cook_simple",
)

EVAL_LAYOUTS = (
    "test_time_simple",
    "test_time_wide",
    "demo_cook_wide",
)

PARTNER_NAMES = ("courier", "potter", "handoff", "noisy")
