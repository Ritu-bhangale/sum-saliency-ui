# models/__init__.py
# ensure this package exposes the registry helpers
from .registry import list_models, get_model, register_model

# import all model wrappers so they register themselves
# Add new model modules here as you create them
try:
    from . import umsi_pp
except Exception:
    # fail loudly in interactive run if you want; for now ignore to allow partial setups
    pass

try:
    from . import deepgaze_pp, pathgan_pp, umsi_pp
except Exception:
    pass