# models/deepgaze_wrapper.py
import numpy as np
from PIL import Image
import os
from .registry import register

WEIGHT_PATH = os.path.join("model_zoo", "scanpath", "DeepGaze++", "centerbias_mit1003.npy")

@register("deepgaze_centerbias")
def predict_deepgaze(img: Image.Image, condition: int) -> np.ndarray:
    """
    Very small wrapper: loads precomputed center-bias map and resizes to input size.
    Returns a float32 saliency map (H x W) normalized 0..1.
    """
    # load and resize
    cb = np.load(WEIGHT_PATH)  # shape depends on file; many centerbias are small like (H, W)
    img_w, img_h = img.size
    # if cb is 1D or not 2D, handle carefully:
    if cb.ndim == 1:
        # fallback: reshape square
        side = int(np.sqrt(cb.size))
        cb = cb.reshape(side, side)
    # normalize
    cb = cb.astype("float32")
    cb = (cb - cb.min()) / (cb.max() - cb.min() + 1e-12)
    # resize to image size using PIL for correct filtering
    from PIL import Image as PILImage
    cb_img = PILImage.fromarray((cb * 255).astype("uint8"))
    cb_resized = cb_img.resize((img_w, img_h), resample=PILImage.BILINEAR)
    out = np.array(cb_resized).astype("float32") / 255.0
    return out
