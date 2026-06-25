import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
from PIL import Image
from model import UCDNet

# Initialize app
app = FastAPI(title="UCDNet Change Detection API")

# Load model once at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = UCDNet(in_channels=13, num_classes=2).to(device)
model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), '..', 'model', 'ucdnet_best.pth'),
    map_location=device
))
model.eval()
print("Model loaded on:", device)


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    arr    = np.load(io.BytesIO(file_bytes))
    arr    = arr.astype(np.float32)
    tensor = torch.tensor(arr).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    return tensor.to(device)


@app.get("/")
def root():
    return {"message": "UCDNet Change Detection API is running"}


@app.post("/predict")
async def predict(
    img1: UploadFile = File(...),
    img2: UploadFile = File(...)
):
    try:
        img1_bytes = await img1.read()
        img2_bytes = await img2.read()

        tensor1 = preprocess_image(img1_bytes)
        tensor2 = preprocess_image(img2_bytes)

        with torch.no_grad():
            output = model(tensor1, tensor2)
            pred   = torch.argmax(output, dim=1)
            pred   = pred.squeeze(0).cpu().numpy()

        change_pixels = int(pred.sum())
        total_pixels  = int(pred.size)
        change_ratio  = round(float(change_pixels / total_pixels) * 100, 2)

        return JSONResponse({
            "status"         : "success",
            "change_mask"    : pred.tolist(),
            "change_pixels"  : change_pixels,
            "total_pixels"   : total_pixels,
            "change_ratio_%" : change_ratio
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)