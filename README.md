````markdown
# 🎭 MediaPipe + LBPH Face Recognition

A real-time face recognition system using **MediaPipe** for face detection and **LBPH** for recognition.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install opencv-contrib-python mediapipe numpy scikit-learn
```
````

### 2. Collect Face Data

```bash
# Collect images for Person 1
python 01_create_dataset_mediapipe.py
# Enter ID: John

# Collect images for Person 2
python 01_create_dataset_mediapipe.py
# Enter ID: Mary

# Collect 50-100+ images per person
```

### 3. Train Model

```bash
python 03_train_model_lbph_mediapipe.py --val-split 0.2
```

### 4. Run Recognition

```bash
# Basic recognition
python 04_predict.py

# With face mesh visualization
python 04_predict.py --show-mesh

# On a single image
python 04_predict.py --image photo.jpg
```

---

## 📋 Pipeline

```
1. Create Dataset  → Collect face images with MediaPipe detection
2. Review Dataset  → Clean up poor quality images (optional)
3. Train Model     → Train LBPH recognizer
4. Recognize       → Real-time face recognition
```

---

## 🎮 Controls

### During Dataset Collection:

- `q` – Quit

### During Recognition:

- `q` – Quit
- `m` – Toggle face mesh overlay

### During Dataset Review:

- `←/→` or `p/n` – Navigate
- `d` – Delete image
- `q` – Quit

---

## ⚙️ Configuration

### Recognition Threshold

```bash
# Stricter matching (lower threshold)
python 04_predict.py --threshold 50

# More lenient matching (higher threshold)
python 04_predict.py --threshold 80
```

### Camera Selection

```bash
# Use different camera
python 04_predict.py --camera 1
```

---

## 📁 Project Structure

```
face_recognition_mediapipe/
├── 01_create_dataset.py    # Collect face images
├── 02_review_dataset.py    # Review and clean dataset
├── 03_train_model.py  # Train LBPH model
├── 04_predict.py                      # Real-time
├── dataset/                           # Training   images
│   ├── John/*.jpg
│   └── Mary/*.jpg
└── models/                            # Trained models
    ├── lbph_face_model.yml
    └── lbph_label_map.pkl
```

---

## ✨ Features

- 🎯 **MediaPipe Detection** – Accurate ML-based face detection
- 🕸️ **468-Point Face Mesh** – Visual landmark overlay
- ⚡ **Real-time Processing** – Live webcam recognition
- 👥 **Multi-Person Support** – Recognize multiple faces
- 🎨 **Interactive Visualization** – Toggle face mesh on/off

---

## 💡 Tips

**For Best Results:**

- Collect 50-100+ images per person
- Use good lighting
- Try different angles and expressions
- Position yourself 2-3 feet from camera

**Troubleshooting:**

- If accuracy is low, collect more training images
- If too many false positives, lower the threshold
- If missing detections, raise MediaPipe confidence

---

## 📝 Requirements

```txt
opencv-contrib-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

---

## 🔬 How It Works

1. **MediaPipe** detects faces and extracts 468 facial landmarks
2. **LBPH** analyzes local texture patterns to identify faces
3. Lower confidence score = better match
4. Real-time processing with visual feedback

---

## 📄 License

MIT License
