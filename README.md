# SUM – Saliency Heatmap Generator

Generate neural saliency heatmaps from UI screenshots to evaluate visual focus, hierarchy, and attention patterns.

---

## 🔍 What this project does

This tool analyzes UI screens and produces saliency heatmaps that highlight areas likely to draw user attention.

📌 Useful for UX designers to validate:

- Visual hierarchy strength  
- CTA prominence and discoverability  
- Balance of layout and whitespace  
- UI clarity & readability  

---

## 🧠 Model

- Architecture: **SUM (Saliency-based UI Model)**  
- Built using **Mamba SS2D + encoder–decoder structure**  
- Supports *conditional* attention-based predictions  
- Currently running on **MPS (Apple Silicon)**  
- CUDA selective scan not enabled (planned for Linux GPU build)

> Goal: Reach research-quality attention maps with fine-tuning + better training data.

---

## 📦 Dataset

_Current dataset is minimal — project is tuned for expansion._

| Stage | Status |
|---|---|
| Base pretrained model | ✔ Loaded |
| UI-specific dataset | 🔄 Required |
| Fine-tuning | 🔥 Planned |

Next steps → Build a dataset of UI screens + annotated fixation regions to train a stronger model.

---

## ⚙ Tech Stack

| Layer | Technology |
|---|---|
| Backend / Inference API | **FastAPI + PyTorch** |
| Heatmap Post-Processing | **OpenCV + Matplotlib** |
| Frontend Viewer | **React + Vite** |
| Hardware Runtime | Apple Silicon (MPS) |


