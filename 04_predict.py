#!/usr/bin/env python3
"""
Real-time Face Recognition using MediaPipe
Works with both single-person and multi-person models
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Paths
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "face_recognition_model.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# CLI
ap = argparse.ArgumentParser(description="MediaPipe Face Recognition")
ap.add_argument("--camera", type=int, default=0, help="Camera index")
ap.add_argument("--image", type=str, help="Run on single image instead of webcam")
ap.add_argument("--confidence", type=float, default=0.7, 
                help="Min detection confidence (0-1)")
ap.add_argument("--threshold", type=float, default=0.5,
                help="Recognition probability threshold for multi-person (0-1)")
args = ap.parse_args()

# Load model
if not MODEL_PATH.exists():
    sys.exit(f"[ERROR] Model not found: {MODEL_PATH}\nRun: python 03_train_model.py")
if not ENCODER_PATH.exists():
    sys.exit(f"[ERROR] Encoder not found: {ENCODER_PATH}")

print("[INFO] Loading model...")
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

# Handle both old and new model formats
if isinstance(model_data, dict):
    model = model_data['model']
    mode = model_data.get('mode', 'multi_person')
else:
    model = model_data
    mode = 'multi_person'

print("[INFO] Loading label encoder...")
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

print(f"[INFO] Mode: {mode.upper().replace('_', ' ')}")
print(f"[INFO] Classes: {', '.join(label_encoder.classes_)}")

def get_embedding(face_img):
    """Generate embedding from face (same as training)"""
    face_resized = cv2.resize(face_img, (160, 160))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) if len(face_resized.shape) == 3 else face_resized
    
    # Histogram features
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    
    # Spatial features
    grid_size = 4
    h, w = gray.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    spatial_features = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            spatial_features.append(cell.mean())
            spatial_features.append(cell.std())
    
    embedding = np.concatenate([hist, spatial_features])
    
    if len(embedding) < 128:
        embedding = np.pad(embedding, (0, 128 - len(embedding)), 'constant')
    else:
        embedding = embedding[:128]
    
    embedding = embedding / (np.linalg.norm(embedding) + 1e-7)
    return embedding

def recognize_faces(frame, face_detection, face_mesh, show_confidence=True):
    """Detect and recognize faces in frame with MediaPipe mesh visualization"""
    frame_h, frame_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Face detection
    detection_results = face_detection.process(rgb_frame)
    
    # Face mesh
    mesh_results = face_mesh.process(rgb_frame)
    
    recognized_count = 0
    
    # Draw MediaPipe Face Mesh first (background)
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            # Draw tesselation
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Draw contours
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    
    # Process detections for recognition
    if detection_results.detections:
        for detection in detection_results.detections:
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            x = max(0, int(bboxC.xmin * frame_w))
            y = max(0, int(bboxC.ymin * frame_h))
            w = int(bboxC.width * frame_w)
            h = int(bboxC.height * frame_h)
            
            # Ensure valid crop
            if w <= 0 or h <= 0:
                continue
            
            x2 = min(frame_w, x + w)
            y2 = min(frame_h, y + h)
            face = frame[y:y2, x:x2]
            
            if face.size == 0:
                continue
            
            try:
                # Generate embedding
                embedding = get_embedding(face)
                embedding = embedding.reshape(1, -1)
                
                # Predict based on mode
                if mode == 'single_person':
                    # One-Class SVM: +1 = match, -1 = unknown
                    prediction = model.predict(embedding)[0]
                    
                    if prediction == 1:
                        name = label_encoder.classes_[0]
                        confidence = 0.95  # High confidence for match
                        color = (0, 255, 0)
                        recognized_count += 1
                    else:
                        name = "Unknown"
                        confidence = 0.0
                        color = (0, 165, 255)
                
                else:  # multi_person mode
                    # Multi-class SVM
                    proba = model.predict_proba(embedding)[0]
                    max_proba = proba.max()
                    pred_idx = proba.argmax()
                    confidence = max_proba
                    
                    if max_proba >= args.threshold:
                        name = label_encoder.classes_[pred_idx]
                        color = (0, 255, 0)
                        recognized_count += 1
                    else:
                        name = "Unknown"
                        color = (0, 165, 255)
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                
                # Draw label background
                label = f"{name}"
                if show_confidence:
                    label += f" ({confidence*100:.1f}%)"
                
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                label_y = max(y - 10, label_h + 10)
                cv2.rectangle(frame, (x, label_y - label_h - 10),
                             (x + label_w + 10, label_y + baseline),
                             color, -1)
                
                cv2.putText(frame, label, (x + 5, label_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                           2, cv2.LINE_AA)
                
            except Exception as e:
                print(f"[WARN] Recognition error: {e}")
    
    return frame, recognized_count

# Single image mode
if args.image:
    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"[ERROR] Image not found: {img_path}")
    
    frame = cv2.imread(str(img_path))
    if frame is None:
        sys.exit(f"[ERROR] Could not read: {img_path}")
    
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=args.confidence
    ) as face_detection, mp_face_mesh.FaceMesh(
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        result, count = recognize_faces(frame, face_detection, face_mesh)
    
    cv2.putText(result, f"Recognized: {count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow(f"Recognition - {img_path.name}", result)
    print(f"[INFO] Recognized {count} face(s)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(0)

# Webcam mode
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    sys.exit("[ERROR] Could not open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[INFO] Starting webcam recognition...")
print(f"[INFO] Detection confidence: {args.confidence}")
if mode == 'multi_person':
    print(f"[INFO] Recognition threshold: {args.threshold}")
print("[INFO] Press 'q' to quit")

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=args.confidence
) as face_detection, mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    fps = 0.0
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame")
            break
        
        # Mirror for natural view
        frame = cv2.flip(frame, 1)
        
        # Recognize faces
        result, count = recognize_faces(frame, face_detection, face_mesh, show_confidence=True)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time
        
        # Draw HUD
        frame_h, frame_w = result.shape[:2]
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
        
        mode_display = "Verification" if mode == 'single_person' else "Recognition"
        hud = f"MediaPipe {mode_display} | Detected: {count}  |  FPS: {fps:.1f}  |  'q' to quit"
        cv2.putText(result, hud, (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add MediaPipe branding
        cv2.putText(result, "MediaPipe", (frame_w - 140, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("MediaPipe Face Recognition", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Recognition stopped")