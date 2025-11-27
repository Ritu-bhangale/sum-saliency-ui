# models/umsi_pp.py
from PIL import Image
import numpy as np
from .registry import register_model

# lazy load holder
_model = None

@register_model("UMSI++")
def umsi_pp(pil_img: Image.Image, condition: int):
    """
    Simple wrapper: resize -> model.predict -> return numpy saliency map
    Edit preprocessing & model loading per the model's requirements.
    """
    global _model
    if _model is None:
        # Example for keras .hdf5; replace with correct loader if different.
        try:
            from tensorflow.keras.models import load_model
        except Exception as e:
            raise RuntimeError("TensorFlow/Keras not available in env. Install or adapt loader.") from e
        _model = load_model("model_zoo/saliency_models/UMSI++/umsi++.hdf5")

    # preprocessing — adapt to how UMSI++ expects input
    img = pil_img.convert("RGB").resize((256, 256))
    arr = np.array(img).astype("float32") / 255.0
    # channel-first or channel-last? adapt if needed:
    if _model.input_shape and len(_model.input_shape) == 4 and _model.input_shape[-1] == 3:
        inp = arr[None, ...]               # NHWC
    else:
        inp = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW

    pred = _model.predict(inp)
    sal = pred.squeeze()
    # normalize to 0..1
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-12)
    return sal
