# models/deepgaze_pp.py
"""
DeepGaze++ wrapper.
- If a full Keras model (deepgaze_model.h5) is present in model_zoo/scanpath_models/DeepGaze++/
  it will be used to predict saliency maps.
- Otherwise, falls back to centerbias_mit1003.npy (already included), resized to the input image.
"""

import os
import numpy as np
from PIL import Image
import logging

LOG = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "model_zoo", "scanpath", "DeepGaze++")
CENTERBIAS_PATH = os.path.join(MODEL_DIR, "centerbias_mit1003.npy")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "deepgaze_model.h5")

# optional TF/Keras import
try:
    # Import lazily so environments without TF still work for centerbias fallback
    from tensorflow import keras  # type: ignore
    HAS_KERAS = True
except Exception:
    HAS_KERAS = False

def _load_centerbias():
    if not os.path.exists(CENTERBIAS_PATH):
        raise FileNotFoundError(f"centerbias file not found at {CENTERBIAS_PATH}")
    arr = np.load(CENTERBIAS_PATH)
    # centerbias might be stored as (H, W) or (1,H,W); normalize to (H, W)
    if arr.ndim == 3:
        arr = arr.squeeze()
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

class DeepGazePP:
    def __init__(self):
        self.model = None
        if HAS_KERAS and os.path.exists(KERAS_MODEL_PATH):
            try:
                LOG.info("Loading DeepGaze++ Keras model from %s", KERAS_MODEL_PATH)
                self.model = keras.models.load_model(KERAS_MODEL_PATH, compile=False)
                LOG.info("DeepGaze++ Keras model loaded.")
            except Exception as e:
                LOG.warning("Failed to load Keras DeepGaze model: %s. Falling back to centerbias.", e)
                self.model = None

        # always load centerbias as fallback/baseline
        try:
            self.centerbias = _load_centerbias()
        except Exception as e:
            LOG.error("Failed to load centerbias: %s", e)
            self.centerbias = None

    def predict_from_pil(self, img_pil, condition=None):
        """
        img_pil: PIL.Image
        condition: ignored (keeps API consistent)
        returns: numpy array (H, W) float32 normalized [0,1]
        """
        img = img_pil.convert("RGB")
        w, h = img.size

        # if keras model exists, try to use it
        if self.model is not None and HAS_KERAS:
            try:
                # Basic preprocessing - adapt as needed for model specifics
                x = img.resize((224, 224))  # many saliency models use ~224, adapt if needed
                x = np.asarray(x).astype(np.float32) / 255.0
                x = np.expand_dims(x, 0)
                pred = self.model.predict(x)
                pred = np.squeeze(pred)
                # resize back to original
                pred_img = Image.fromarray((pred * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
                out = np.asarray(pred_img).astype(np.float32) / 255.0
                # if multi-channel, collapse
                if out.ndim == 3:
                    out = out[..., 0]
                out = out - out.min()
                if out.max() > 0:
                    out /= out.max()
                return out.astype(np.float32)
            except Exception as e:
                LOG.warning("Keras model prediction failed: %s - falling back to centerbias", e)

        # fallback: resize centerbias to image size
        if self.centerbias is None:
            raise RuntimeError("No DeepGaze model or centerbias available.")
        cb = Image.fromarray((self.centerbias * 255).astype(np.uint8))
        cb = cb.resize((w, h), Image.BILINEAR)
        out = np.asarray(cb).astype(np.float32) / 255.0
        out = out - out.min()
        if out.max() > 0:
            out /= out.max()
        return out.astype(np.float32)

# convenience singleton for registry
_default = DeepGazePP()

def predict_from_pil(img_pil, condition=None):
    return _default.predict_from_pil(img_pil, condition)
