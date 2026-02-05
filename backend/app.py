from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io

# ------------------------
# FastAPI setup
# ------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# MNIST Model Definition
# ------------------------
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------
# Load trained model
# ------------------------
model = MNISTModel()
model.load_state_dict(torch.load("mnist_model.pt", map_location="cpu"))
model.eval()

# ------------------------
# Prediction endpoint
# ------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))

    # Convert to numpy
    image_array = np.array(image).astype(np.float32) / 255.0

    # Save BEFORE normalization
    Image.fromarray((image_array * 255).astype(np.uint8)).save(
        "debug_before_norm.png"
    )

    # Normalize
    image_array = (image_array - 0.1307) / 0.3081

    # Save AFTER normalization (rescaled for viewing)
    debug_vis = (image_array * 0.3081) + 0.1307
    debug_vis = np.clip(debug_vis, 0, 1)
    Image.fromarray((debug_vis * 255).astype(np.uint8)).save(
        "debug_after_norm.png"
    )

    tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()

    return {"prediction": prediction}
