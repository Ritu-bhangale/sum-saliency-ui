import io
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_and_preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    orig_size = image.size
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image)
    return image, orig_size


def predict_saliency_map(img, condition, model, device):
    img = img.unsqueeze(0).to(device)
    one_hot_condition = torch.zeros((1, 4), device=device)
    one_hot_condition[0, condition] = 1
    model.eval()
    with torch.no_grad():
        pred_saliency = model(img, one_hot_condition)

    pred_saliency = pred_saliency.squeeze().cpu().numpy()
    return pred_saliency


def overlay_heatmap_on_image(original_img_path, heatmap_img_path, output_img_path):
    """
    Create a sparse, spot-like overlay similar to online tools:
    - keep only top X% saliency
    - blur to get blobs
    - blend with per-pixel alpha
    """
    # Read original
    orig = cv2.imread(original_img_path)
    if orig is None:
        raise RuntimeError(f"Could not read original image: {original_img_path}")
    H, W = orig.shape[:2]

    # Read grayscale heatmap (written by write_heatmap_to_image)
    hmap = cv2.imread(heatmap_img_path, cv2.IMREAD_GRAYSCALE)
    if hmap is None:
        raise RuntimeError(f"Could not read heatmap image: {heatmap_img_path}")

    hmap = cv2.resize(hmap, (W, H), interpolation=cv2.INTER_AREA)
    hmap = hmap.astype(np.float32) / 255.0

    # Keep only top X% of saliency
    # Try 0.85–0.9; lower = more area highlighted
    keep_top = 0.88
    thr = np.quantile(hmap, keep_top)
    mask = np.clip((hmap - thr) / (1.0 - thr + 1e-6), 0.0, 1.0)

    # Smooth to get nicer blobs
    mask = cv2.GaussianBlur(mask, (41, 41), 0)

    # Optional extra sharpening of hotspots
    mask = np.power(mask, 1.2)

    # Convert mask to color map
    mask_uint8 = (mask * 255).astype(np.uint8)
    color = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)

    # Normalize to [0,1] for blending
    orig_f = orig.astype(np.float32) / 255.0
    color_f = color.astype(np.float32) / 255.0

    # Per-pixel alpha: strong saliency = more color
    strength = 0.85  # global max opacity of heatmap
    alpha = (mask * strength)[..., None]  # H x W x 1

    out = orig_f * (1.0 - alpha) + color_f * alpha
    out = (out * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite(output_img_path, out)


def write_heatmap_to_image(heatmap, orig_size, output_path):
    """
    Save a sparse, blob-like heatmap (no overlay here yet).
    This image is just an intermediate; the nice overlay is done later.
    """
    # Normalize to [0, 1]
    heatmap = heatmap - heatmap.min()
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val

    # Optional sharpening of peaks
    gamma = 1.5  # >1 makes peaks sharper
    heatmap = np.power(heatmap, gamma)

    # Resize to original size (width, height)
    w, h = orig_size
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_AREA)

    # Save as grayscale for later processing
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    cv2.imwrite(output_path, heatmap_uint8)
