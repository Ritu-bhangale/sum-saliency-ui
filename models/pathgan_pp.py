# models/pathgan_pp.py
"""
PathGAN++ wrapper.
- Loads the Keras generator if present at model_zoo/scanpath_models/PathGAN++/generator_PathGAN++.h5
- If generator is missing, raises a FileNotFoundError so caller knows it's not available.
"""

import os
import numpy as np
from PIL import Image
import logging

LOG = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "model_zoo", "scanpath", "PathGAN++")
GENERATOR_PATH = os.path.join(MODEL_DIR, "generator_PathGAN++.h5")

# optional TF/Keras import
try:
    from tensorflow import keras  # type: ignore
    HAS_KERAS = True
except Exception:
    HAS_KERAS = False

class PathGANPP:
    def __init__(self):
        self.gen = None
        if not HAS_KERAS:
            LOG.warning("TensorFlow/Keras not available in environment. PathGAN++ cannot be used.")
            return
        if not os.path.exists(GENERATOR_PATH):
            LOG.info("PathGAN++ generator not found at %s", GENERATOR_PATH)
            return
        try:
            LOG.info("Loading PathGAN++ generator from %s", GENERATOR_PATH)
            self.gen = keras.models.load_model(GENERATOR_PATH, compile=False)
            LOG.info("PathGAN++ generator loaded.")
        except Exception as e:
            LOG.exception("Failed to load PathGAN++ generator: %s", e)
            self.gen = None

    def predict_from_pil(self, img_pil, condition=None):
        """
        img_pil: PIL.Image
        condition: may be used by multi-condition variants; ignored here
        returns: numpy array (H, W) float32 normalized [0,1]
        """
        if self.gen is None:
            raise RuntimeError(f"PathGAN++ generator not loaded. Put generator at: {GENERATOR_PATH}")

        # Preprocess - adapt to the generator's expected input shape
        w, h = img_pil.size
        # common PathGAN setups use 256x256 or 224; change if you know exact size
        in_w, in_h = 256, 256
        x = img_pil.convert("RGB").resize((in_w, in_h))
        x = np.asarray(x).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
        # model predict -> likely single-channel map
        pred = self.gen.predict(x)
        pred = np.squeeze(pred)
        # if multi-channel, collapse
        if pred.ndim == 3:
            pred = pred[..., 0]
        # resize back to original
        pred_img = Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        out = np.asarray(pred_img).astype(np.float32) / 255.0
        out = out - out.min()
        if out.max() > 0:
            out /= out.max()
        return out.astype(np.float32)

_default = PathGANPP()

def predict_from_pil(img_pil, condition=None):
    return _default.predict_from_pil(img_pil, condition)
