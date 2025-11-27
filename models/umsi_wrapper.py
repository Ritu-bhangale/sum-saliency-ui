# models/umsi_wrapper.py
import os
from PIL import Image
import numpy as np
from .registry import register

UMSI_PATH = os.path.join("model_zoo", "saliency", "UMSI++", "umsi++.hdf5")

_MODEL = None
def _load_umsi():
    # Try keras load first
    try:
        from tensorflow.keras.models import load_model
        return load_model(UMSI_PATH, compile=False)
    except Exception as e:
        # If it's not a pure Keras model, user may have custom loader. Raise a helpful error.
        raise RuntimeError(f"Failed to load UMSI model via keras.load_model: {e}\n"
                           "If UMSI++ uses a custom architecture, add a loader in models/umsi_wrapper.py")

@register("umsi_plus")
def predict_umsi(img: Image.Image, condition: int) -> np.ndarray:
    """
    Load UMSI++ model and run inference. Returns saliency map HxW normalized.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_umsi()

    img = img.convert("RGB")
    # UMSI models typically expect 256x256 or 384x384 — check paper / weights. We'll use 256 as safe default.
    target_size = (256, 256)
    img_resized = img.resize(target_size)
    x = np.array(img_resized).astype("float32") / 255.0
    x = np.expand_dims(x, 0)
    pred = _MODEL.predict(x)
    pred = np.squeeze(pred)
    # reduce to single channel and normalize
    if pred.ndim == 3:
        pred = pred.mean(axis=-1)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-12)

    # resize back to original image size
    from PIL import Image as PILImage
    pimg = PILImage.fromarray((pred * 255).astype("uint8"))
    pimg = pimg.resize(img.size, resample=PILImage.BILINEAR)
    out = np.array(pimg).astype("float32") / 255.0
    return out
