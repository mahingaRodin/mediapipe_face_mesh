# MEDIAPIPE Face Recognition

## Pipeline Overview
1️⃣ **Create Dataset** – Detect a face, and save face images  
2️⃣ **Review Dataset** – Delete poor or blurred images  
3️⃣ **Train Model** – Learn local texture patterns with LBPH  
4️⃣ **Recognize Faces** – Predict names from live webcam input  

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt