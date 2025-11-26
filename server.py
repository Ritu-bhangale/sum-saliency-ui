# server.py
import os
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


from inference import (
    setup_model,
    load_and_preprocess_image,
    predict_saliency_map,
    write_heatmap_to_image,
    overlay_heatmap_on_image,
)

# ---------- Setup ----------

app = FastAPI()

# allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # in prod, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

model = setup_model(device)
print("Model loaded successfully")


# ---------- Routes ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    condition: int = Form(...),              # 0–3
    file: UploadFile = File(...),
):
    if condition not in (0, 1, 2, 3):
        return JSONResponse(
            {"error": "condition must be 0, 1, 2, or 3"},
            status_code=400,
        )

    # save uploaded image
    uid = uuid.uuid4().hex
    suffix = Path(file.filename).suffix or ".png"
    input_path = TMP_DIR / f"{uid}{suffix}"
    heatmap_path = TMP_DIR / f"{uid}_saliencymap.png"
    overlay_path = TMP_DIR / f"{uid}_overlay.png"

    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    # run model
    img_tensor, orig_size = load_and_preprocess_image(str(input_path))
    saliency = predict_saliency_map(img_tensor, condition, model, device)
    write_heatmap_to_image(saliency, orig_size, str(heatmap_path))
    overlay_heatmap_on_image(str(input_path), str(heatmap_path), str(overlay_path))

    # return overlay PNG
    return FileResponse(
        path=str(overlay_path),
        media_type="image/png",
        filename="overlay.png",
    )
