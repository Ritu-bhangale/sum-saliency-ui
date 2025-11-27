# models/registry.py
from typing import Callable, Dict, List

MODEL_REGISTRY: Dict[str, Callable] = {}

def register_model(name: str):
    """Decorator to register a model factory/wrapper function."""
    def decorator(fn: Callable):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator

def list_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())

def get_model(name: str) -> Callable:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model {name!r} not registered. Available: {list_models()}")
    return MODEL_REGISTRY[name]
