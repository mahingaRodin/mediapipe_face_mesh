#!/usr/bin/env python3
"""
Real-time Face Recognition using MediaPipe + LBPH
MediaPipe: Face detection and mesh visualization
LBPH: Face recognition algorithm
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
MODEL_PATH = MODELS_DIR / "lbph_face_model.yml"
LABEL_MAP_PATH = MODELS_DIR / "lbph_label_map.pkl"

# CLI
ap = argparse.ArgumentParser(description="MediaPipe + LBPH Face Recognition")
ap.add_argument("--camera", type=int, default=0, help="Camera index")
ap.add_argument("--image", type=str, help="Run on single image")
ap.add_argument("--confidence", type=float, default=0.7, 
                help="Min MediaPipe detection confidence (0-1)")
ap.add_argument("--threshold", type=float, default=70.0,
                help="LBPH confidence threshold (lower=stricter, default: 70)")
ap.add_argument("--show-mesh", action="store_true",
                help="Show MediaPipe face mesh overlay")
args = ap.parse_args()

# Load models
if not MODEL_PATH.exists():
    sys.exit(f"[ERROR] Model not found: {MODEL_PATH}\nRun: python 03_train_model_lbph_mediapipe.py")
if not LABEL_MAP_PATH.exists():
    sys.exit(f"[ERROR] Label map not found: {LABEL_MAP_PATH}")

print("[INFO] Loading LBPH model...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

print("[INFO] Loading label mapping...")
with open(LABEL_MAP_PATH, 'rb') as f:
    label_map = pickle.load(f)

# Create reverse mapping (label_id -> name)
reverse_map = {v: k for k, v in label_map.items()}

print(f"[INFO] Technology: MediaPipe (detection) + LBPH (recognition)")
print(f"[INFO] Recognized people: {', '.join(sorted(label_map.keys()))}")
print(f"[INFO] LBPH threshold: {args.threshold:.1f} (lower=stricter)")

def recognize_face(face_img):
    """Recognize face using LBPH"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        
        # Resize to training size
        gray_resized = cv2.resize(gray, (160, 160))
        
        # Predict
        label_id, confidence = recognizer.predict(gray_resized)
        
        # LBPH confidence: lower values = better match
        if confidence < args.threshold:
            name = reverse_map.get(label_id, "Unknown")
            is_recognized = True
        else:
            name = "Unknown"
            is_recognized = False
        
        return name, confidence, is_recognized
    except Exception as e:
        print(f"[ERROR] Recognition failed: {e}")
        return "Error", 999, False

def recognize_faces(frame, face_detection, face_mesh, show_mesh=True):
    """Detect and recognize faces with MediaPipe + LBPH"""
    frame_h, frame_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe face detection
    detection_results = face_detection.process(rgb_frame)
    
    # MediaPipe face mesh (optional)
    mesh_results = None
    if show_mesh:
        mesh_results = face_mesh.process(rgb_frame)
    
    recognized_count = 0
    
    # Draw face mesh (background layer)
    if show_mesh and mesh_results and mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            # Tesselation
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Contours
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    
    # Process face detections for recognition
    if detection_results.detections:
        for detection in detection_results.detections:
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            x = max(0, int(bboxC.xmin * frame_w))
            y = max(0, int(bboxC.ymin * frame_h))
            w = int(bboxC.width * frame_w)
            h = int(bboxC.height * frame_h)
            
            # Add padding for better recognition
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame_w - x, w + 2 * padding)
            h = min(frame_h - y, h + 2 * padding)
            
            x2 = min(frame_w, x + w)
            y2 = min(frame_h, y + h)
            
            if w <= 0 or h <= 0:
                continue
            
            face = frame[y:y2, x:x2]
            
            if face.size == 0:
                continue
            
            try:
                # Recognize using LBPH
                name, confidence, is_recognized = recognize_face(face)
                
                # Color based on recognition
                if is_recognized:
                    color = (0, 255, 0)  # Green
                    recognized_count += 1
                else:
                    color = (0, 165, 255)  # Orange
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                
                # Draw label
                label = f"{name} ({confidence:.1f})"
                
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
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        result, count = recognize_faces(frame, face_detection, face_mesh, show_mesh=args.show_mesh)
    
    cv2.putText(result, f"Recognized: {count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow(f"MediaPipe + LBPH Recognition", result)
    print(f"[INFO] Recognized {count} face(s)")
    print("[INFO] Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(0)

# Webcam mode
print("\n" + "="*60)
print("[INFO] Starting webcam recognition...")
print(f"[INFO] MediaPipe detection confidence: {args.confidence}")
print(f"[INFO] LBPH recognition threshold: {args.threshold:.1f}")
print(f"[INFO] Face mesh overlay: {'Enabled' if args.show_mesh else 'Disabled'}")
print("[INFO] Press 'q' to quit, 'm' to toggle mesh")
print("="*60 + "\n")

# Try to open camera
print("[INFO] Opening camera...")
cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # Windows-specific backend

if not cap.isOpened():
    print("[WARN] DSHOW backend failed, trying default...")
    cap = cv2.VideoCapture(args.camera)

if not cap.isOpened():
    sys.exit("[ERROR] Could not open camera. Check permissions and camera connection.")

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Verify camera is working
ret, test_frame = cap.read()
if not ret or test_frame is None:
    cap.release()
    sys.exit("[ERROR] Camera opened but cannot read frames. Check camera access.")

print(f"[INFO] Camera opened successfully!")
print(f"[INFO] Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

show_mesh = args.show_mesh
frame_count = 0

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=args.confidence
) as face_detection, mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    fps = 0.0
    prev_time = time.time()
    
    print("[INFO] Camera window should appear now...")
    print("[INFO] If no window appears, check your display settings")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        frame_count += 1
        
        # Mirror for natural view
        frame = cv2.flip(frame, 1)
        
        # Recognize faces
        try:
            result, count = recognize_faces(frame, face_detection, face_mesh, show_mesh=show_mesh)
        except Exception as e:
            print(f"[ERROR] Processing error: {e}")
            result = frame
            count = 0
        
        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time
        
        # Draw HUD
        frame_h, frame_w = result.shape[:2]
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
        
        mesh_status = "ON" if show_mesh else "OFF"
        hud = f"MediaPipe+LBPH | Recognized: {count} | FPS: {fps:.1f} | Mesh: {mesh_status} | 'q' quit, 'm' toggle"
        cv2.putText(result, hud, (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show frame
        cv2.imshow("MediaPipe + LBPH Face Recognition", result)
        
        # First frame notification
        if frame_count == 1:
            print(f"[INFO] ✓ Window displayed! Frame size: {frame_w}x{frame_h}")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Quit key pressed")
            break
        elif key == ord('m'):
            show_mesh = not show_mesh
            print(f"[INFO] Face mesh: {'Enabled' if show_mesh else 'Disabled'}")

cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Recognition stopped")
print(f"[INFO] Total frames processed: {frame_count}")