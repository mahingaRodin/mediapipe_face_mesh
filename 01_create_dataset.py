#!/usr/bin/env python3
"""
Dataset Creator using MediaPipe Face Detection & Mesh
Captures face images from webcam with real-time detection and face mesh visualization
"""
import cv2
import time
import os
import mediapipe as mp

# ---- Config ----
MAX_IMAGES = 250
INTERVAL_MS = 500
DATASET_DIR = 'dataset'

# ---- Setup MediaPipe ----
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

os.makedirs(DATASET_DIR, exist_ok=True)

ID = input('Enter your ID: ').strip()
print("Please get your face ready!")
time.sleep(2)

# Use default webcam
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 30)

if not cam.isOpened():
    raise SystemExit("[Error] Could not open webcam.")

actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[Info] Webcam initialized at {actual_w}x{actual_h}")

win = "MediaPipe Face Dataset Creator"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1280, 720)

start_time = time.time()
last_capture_ts = start_time
image_count = 0
last_saved_flash_ts = 0.0
FLASH_MS = 200

print("\n[Ready] Position yourself in front of the webcam")
print("        Press 'q' to quit anytime\n")

# Initialize MediaPipe Face Detection and Mesh
with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.7
) as face_detection, mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as face_mesh:
    
    while True:
        ok, frame = cam.read()
        if not ok:
            print("[Warn] Failed to read frame from webcam.")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces for bounding box
        detection_results = face_detection.process(rgb_frame)
        
        # Process face mesh for visualization
        mesh_results = face_mesh.process(rgb_frame)
        
        face_detected = False
        face_bbox = None
        
        if detection_results.detections:
            for detection in detection_results.detections:
                face_detected = True
                
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * frame_w)
                y = int(bboxC.ymin * frame_h)
                w = int(bboxC.width * frame_w)
                h = int(bboxC.height * frame_h)
                
                # Ensure coordinates are within frame
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)
                
                face_bbox = (x, y, w, h)
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw detection confidence
                conf = detection.score[0] if detection.score else 0
                cv2.putText(frame, f"Conf: {conf:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Capture at intervals
                elapsed_ms = (time.time() - last_capture_ts) * 1000.0
                if elapsed_ms >= INTERVAL_MS and image_count < MAX_IMAGES:
                    # Create person folder
                    person_dir = os.path.join(DATASET_DIR, ID)
                    os.makedirs(person_dir, exist_ok=True)
                    
                    # Save face crop
                    face_crop = frame[y:y+h, x:x+w]
                    filename = os.path.join(
                        person_dir, f"{ID}_{int(time.time() * 1000)}.jpg"
                    )
                    cv2.imwrite(filename, face_crop)
                    image_count += 1
                    last_capture_ts = time.time()
                    last_saved_flash_ts = last_capture_ts
                    
                    print(f"✓ Captured [{image_count}/{MAX_IMAGES}]: {os.path.basename(filename)}")
                    break
        
        # Draw MediaPipe Face Mesh (shows we're using MediaPipe!)
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Draw tesselation (mesh)
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
                
                # Draw irises
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        
        # ---- HUD ----
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        status = f"ID: {ID}  |  Captured: {image_count}/{MAX_IMAGES}  |  Interval: {INTERVAL_MS}ms"
        cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # MediaPipe logo/indicator
        cv2.putText(frame, "MediaPipe", (frame_w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Face detection indicator
        if face_detected:
            cv2.circle(frame, (frame_w - 170, 25), 8, (0, 255, 0), -1)
            cv2.putText(frame, "FACE DETECTED", (frame_w - 320, 32), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (frame_w - 170, 25), 8, (0, 0, 255), -1)
            cv2.putText(frame, "NO FACE", (frame_w - 260, 32), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Flash "SAVED!"
        if (time.time() - last_saved_flash_ts) * 1000.0 <= FLASH_MS:
            text = "SAVED!"
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
            )
            x = (frame_w - text_w) // 2
            y = 120
            cv2.rectangle(frame, (x - 20, y - text_h - 15),
                         (x + text_w + 20, y + baseline + 15), (0, 255, 0), -1)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Progress bar
        bar_h = 25
        bar_y = frame_h - bar_h - 10
        bar_w = frame_w - 40
        pct = min(1.0, image_count / float(MAX_IMAGES))
        filled = int(bar_w * pct)
        
        cv2.rectangle(frame, (20, bar_y), (20 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, bar_y), (20 + filled, bar_y + bar_h), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, bar_y), (20 + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        
        pct_text = f"{int(pct * 100)}%"
        cv2.putText(frame, pct_text, (20 + bar_w + 15, bar_y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Instructions
        instruction = "MediaPipe Face Mesh Active | Move head slightly | Press 'Q' to quit"
        cv2.putText(frame, instruction, (20, frame_h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.imshow(win, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or image_count >= MAX_IMAGES:
            break

cam.release()
cv2.destroyAllWindows()

print(f"\n{'='*50}")
print(f"✓ Dataset generation complete!")
print(f"{'='*50}")
print(f"  ID: {ID}")
print(f"  Total Images: {image_count}")
print(f"  Location: '{DATASET_DIR}/{ID}/'")
print(f"  Technology: MediaPipe Face Detection & Mesh")
print(f"{'='*50}\n")