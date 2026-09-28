"""Recipe name -> class registry. Adding a new vertical means adding one
entry here, never touching harness/run_cycle.py."""
from harness.recipe import Recipe

_REGISTRY: dict[str, type] = {}


def register(name: str, recipe_class: type):
    _REGISTRY[name] = recipe_class


def get_recipe(name: str) -> Recipe:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown recipe {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
