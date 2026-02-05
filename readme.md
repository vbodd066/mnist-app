🧠 MNIST Digit Detector

An interactive, full-stack machine learning demo for handwritten digit recognition using the MNIST dataset.
Users draw digits in a browser-based canvas UI, and a trained neural network predicts the digit in real time.

This project demonstrates:
End-to-end ML deployment
Image preprocessing aligned with MNIST
A modern, interactive React frontend
A Python + FastAPI backend for inference

✨ Features

    ✍️ Canvas-based digit drawing
    🔢 One digit per canvas (multiple canvases supported)
    🧠 MNIST-trained neural network
    🎯 Automatic centering, scaling, and padding (MNIST-style preprocessing)
    ⚡ Real-time prediction via FastAPI
    ⌨️ Keyboard shortcuts
        Enter → Predict
        Esc → Clear
    🎨 Polished UI
        Rounded canvases
        Animated glowing title
        Hover-responsive buttons
    🧩 Modular design (easy to extend to more digits or models)

🏗 Project Structure

mnist-digit-detector/
├── backend/
│   ├── app.py              # FastAPI backend
│   ├── train.py            # MNIST training script
│   ├── mnist_model.pt      # Trained model weights
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main UI layout & logic
│   │   ├── DigitCanvas.js  # Reusable canvas component
│   │   └── index.js
│   └── package.json
│
└── README.md

🔧 Tech Stack
Frontend
    React
    HTML5 <canvas>
    Inline CSS styling
    Keyboard + mouse interactions

Backend
    FastAPI
    PyTorch
    Pillow / NumPy
    REST API for inference

Machine Learning
    MNIST dataset
    Fully connected neural network (MLP)

MNIST-style preprocessing:
    Bounding box detection
    Rescaling to 20×20
    Padding to 28×28
    Normalization

🚀 Getting Started

1️⃣ Clone the repository
git clone https://github.com/your-username/mnist-digit-detector.git
cd mnist-digit-detector

2️⃣ Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Run the backend:
uvicorn app:app --reload
Verify it’s running:
http://127.0.0.1:8000/docs

3️⃣ Frontend setup
cd frontend
npm install
npm start
Open:
http://localhost:3000

🧪 How It Works
User draws a digit in each canvas

Frontend:
Detects the drawn digit’s bounding box
Centers and rescales it to match MNIST
Sends a 28×28 grayscale image to the backend

Backend:
Normalizes input
Runs inference with the trained model
Returns the predicted digit
UI displays the result instantly

⌨️ Controls
Action	Input
Predict	Enter key
Clear canvases	Esc key
Draw digit	Mouse / Trackpad

🔮 Future Improvements
    ✅ Switch to a CNN for higher accuracy
    🔢 Support more digit canvases
    📊 Show prediction confidence scores
    📱 Touch/mobile support
    🚀 Deployment (Vercel + Render / Fly.io)
    🧾 Batch inference endpoint

📚 Notes
This project intentionally avoids over-complicated OCR techniques.
The goal is clarity, correctness, and extensibility, not raw benchmark performance.
The UI preprocessing mirrors the MNIST pipeline closely, which is critical for good results. This was the first ML project I have done and is very "hello word" esk.

📄 License
MIT License — free to use, modify, and learn from.