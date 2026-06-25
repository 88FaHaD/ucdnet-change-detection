# UCDNet — Satellite Change Detection

A deep learning web app for detecting man-made changes between two Sentinel-2 satellite images.

## Live Demo
🌐 [Try the app](https://ucdnet-change-detection-demo.streamlit.app/)

> **Note:** First inference may take 30-60 seconds as the server wakes from sleep.

## Model Performance
| Metric | Our Model | Paper (UCDNet) |
|--------|-----------|----------------|
| Accuracy | 98.60% | — |
| F1 Score | 82.37% | 89.21% |
| Jaccard Index (IoU) | 70.03% | 80.53% |
| Parameters | 1,031,282 | — |

## Features
- 53 pre-loaded test patches (no satellite data needed)
- Upload your own 64x64x13 .npy patches
- Visualizes before/after RGB and predicted change mask
- Shows ground truth comparison

## Architecture
- Model: UCDNet (Siamese encoder-decoder)
- Input: 64x64 patches, 13 Sentinel-2 bands
- Cities trained: Mumbai and Abu Dhabi
- Framework: PyTorch

## Stack
- PyTorch (model)
- FastAPI (backend API)
- Streamlit (frontend)
- Render (backend hosting)
- Streamlit Cloud (frontend hosting)

## Run Locally
pip install -r requirements.txt
Terminal 1: cd backend and uvicorn main:app --reload --port 8000
Terminal 2: cd frontend and streamlit run streamlit_app.py

## Reference
UCDNet: Basavaraju et al., IEEE TGRS 2022
DOI: 10.1109/TGRS.2022.3161337
