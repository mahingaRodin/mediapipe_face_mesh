#!/usr/bin/env python3
"""
Dataset Preview and Cleaner with MediaPipe Face Detection
Browse and curate your face dataset with visual face detection overlay
"""
import cv2
import time
import mediapipe as mp
from pathlib import Path
from shutil import rmtree
import numpy as np

# Cleanup old trash directories
for trash_dir in Path("dataset").rglob(".trash"):
    try:
        rmtree(trash_dir, ignore_errors=True)
        print(f"[cleanup] Removed old trash folder: {trash_dir}")
    except Exception as e:
        print(f"[warn] Could not remove {trash_dir}: {e}")

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def preview(folder="dataset", pattern="*.jpg", delay_ms=1000):
    """Preview images with MediaPipe face detection overlay"""
    folder = Path(folder)
    
    files = sorted(
        f for f in folder.rglob(pattern)
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        and not f.name.startswith(".")
    )
    
    if not files:
        print(f"No images found in '{folder}'")
        return
    
    win = "MediaPipe Dataset Preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    
    FRAME_W, FRAME_H = 1280, 720
    TOP_BAR_H, BOTTOM_BAR_H = 55, 55
    VIEW_X, VIEW_Y = 0, TOP_BAR_H
    VIEW_W, VIEW_H = FRAME_W, FRAME_H - TOP_BAR_H - BOTTOM_BAR_H
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    GREEN = (0, 180, 0)
    WHITE = (255, 255, 255)
    PAD_X, PAD_Y = 20, 15
    
    FLASH_TS = 0.0
    FLASH_MS = 400
    FLASH_TEXT = ""
    
    def delete_current(img_path: Path):
        nonlocal FLASH_TS, FLASH_TEXT
        try:
            img_path.unlink(missing_ok=True)
            FLASH_TEXT = "Deleted permanently"
            FLASH_TS = time.time()
            print(f"[deleted] {img_path}")
            return True
        except Exception as e:
            FLASH_TEXT = "Delete failed"
            FLASH_TS = time.time()
            print(f"[error] Could not delete {img_path}: {e}")
            return False
    
    def draw_flash(frame):
        if (time.time() - FLASH_TS) * 1000 <= FLASH_MS and FLASH_TEXT:
            text = FLASH_TEXT
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 2)
            x1, y1 = FRAME_W - PAD_X - tw - 24, 8
            x2, y2 = FRAME_W - PAD_X, 8 + th + 18
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 160, 0), -1)
            cv2.putText(frame, text, (x1 + 12, y2 - 8), 
                       FONT, 0.65, WHITE, 2, cv2.LINE_AA)
    
    def draw_frame(img_path, autoplay, delay, idx, face_detection):
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        
        img = cv2.imread(str(img_path))
        if img is not None and img.size > 0:
            # Detect faces in the image
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_img)
            
            # Draw face detections on original image
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw = img.shape[:2]
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    
                    # Draw on image
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw confidence score
                    score = detection.score[0] if detection.score else 0
                    cv2.putText(img, f"{score:.2f}", (x, y - 10),
                               FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Letterbox the image
            ih, iw = img.shape[:2]
            scale = min(VIEW_W / iw, VIEW_H / ih)
            new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            off_x = VIEW_X + (VIEW_W - new_w) // 2
            off_y = VIEW_Y + (VIEW_H - new_h) // 2
            frame[off_y:off_y+new_h, off_x:off_x+new_w] = resized
        
        # Top bar
        cv2.rectangle(frame, (0, 0), (FRAME_W, TOP_BAR_H), GREEN, -1)
        left_text = f"[{idx+1}/{len(files)}] {img_path.name}"
        state_text = f"  |  {'PLAY' if autoplay else 'PAUSE'}  |  {delay}ms"
        top_text = f"{left_text}{state_text}"
        cv2.putText(frame, top_text, (PAD_X, TOP_BAR_H - PAD_Y),
                   FONT, 0.8, WHITE, 2, cv2.LINE_AA)
        
        # Bottom bar
        cv2.rectangle(frame, (0, FRAME_H - BOTTOM_BAR_H), (FRAME_W, FRAME_H), GREEN, -1)
        help_text = "←/p: prev  →/n: next  space/s: play/pause  +/- speed  d: DELETE  q/ESC: quit"
        cv2.putText(frame, help_text, (PAD_X, FRAME_H - PAD_Y),
                   FONT, 0.65, WHITE, 2, cv2.LINE_AA)
        
        draw_flash(frame)
        return frame
    
    # Main loop with MediaPipe
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_detection:
        
        idx = 0
        autoplay = False
        last = time.time()
        
        while True:
            if not files:
                blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
                cv2.rectangle(blank, (0, 0), (FRAME_W, TOP_BAR_H), GREEN, -1)
                cv2.putText(blank, "No images left. Press q to exit.",
                           (PAD_X, TOP_BAR_H - PAD_Y), FONT, 0.8, WHITE, 2, cv2.LINE_AA)
                cv2.imshow(win, blank)
            else:
                out = draw_frame(files[idx], autoplay, delay_ms, idx, face_detection)
                cv2.imshow(win, out)
            
            if files and autoplay and (time.time() - last) * 1000 >= delay_ms:
                idx = (idx + 1) % len(files)
                last = time.time()
            
            key = cv2.waitKeyEx(30)
            
            KEY_LEFT = 2424832
            KEY_RIGHT = 2555904
            
            if key == -1:
                continue
            
            if key in (ord('q'), ord('Q'), 27):
                break
            elif files and (key in (KEY_RIGHT, ord('n'), ord('N'))):
                idx = (idx + 1) % len(files)
                last = time.time()
            elif files and (key in (KEY_LEFT, ord('p'), ord('P'))):
                idx = (idx - 1) % len(files)
                last = time.time()
            elif key in (ord(' '), ord('s'), ord('S')):
                autoplay = not autoplay
                last = time.time()
            elif key in (ord('+'), ord('=')):
                delay_ms = max(50, int(delay_ms * 0.8))
            elif key in (ord('-'), ord('_')):
                delay_ms = min(5000, int(delay_ms * 1.25))
            elif files and key in (ord('d'), ord('D')):
                to_delete = files[idx]
                if delete_current(to_delete):
                    try:
                        files.pop(idx)
                    except Exception:
                        pass
                    if files:
                        idx %= len(files)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    preview(folder="dataset", pattern="*.jpg", delay_ms=250)