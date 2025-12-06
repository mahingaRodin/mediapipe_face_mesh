#!/usr/bin/env python3
import argparse
import pickle
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATASET = ROOT / "dataset"
MODEL_PKL = MODELS / "lbph_face_model.pkl"
LABEL_MAP_PKL = MODELS / "lbph_label_map.pkl"

# CLI
ap = argparse.ArgumentParser(description="Train LBPH face recognition model")
ap.add_argument("--val-split", type=float, default=0.2,
                help="Validation split (default: 0.2, set to 0 to skip validation)")
ap.add_argument("--radius", type=int, default=1,
                help="LBPH radius parameter (default: 1)")
ap.add_argument("--neighbors", type=int, default=8,
                help="LBPH neighbors parameter (default: 8)")
ap.add_argument("--grid-x", type=int, default=8,
                help="LBPH grid X parameter (default: 8)")
ap.add_argument("--grid-y", type=int, default=8,
                help="LBPH grid Y parameter (default: 8)")
args = ap.parse_args()

def load_dataset(dataset_path):
    """Load pre-cropped face images from dataset folders"""
    faces = []
    labels = []
    label_names = []
    image_paths = []
    
    print(f"\n[INFO] Loading dataset from {dataset_path}")
    
    # Iterate through person folders
    person_dirs = [d for d in sorted(dataset_path.iterdir()) 
                  if d.is_dir() and not d.name.startswith('.')]
    
    if not person_dirs:
        print(f"[ERROR] No person folders found in {dataset_path}")
        print("[INFO] Expected structure: dataset/person_name/*.jpg")
        return [], [], [], []
    
    # Create label mapping
    label_map = {person_dir.name: idx for idx, person_dir in enumerate(person_dirs)}
    
    for person_dir in person_dirs:
        person_name = person_dir.name
        person_label = label_map[person_name]
        print(f"[INFO] Processing {person_name} (label: {person_label})...")
        
        img_count = 0
        # Load images for this person
        for img_path in sorted(person_dir.glob('*')):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            try:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    print(f"[WARN] Could not read {img_path.name}")
                    continue
                
                # Convert to grayscale (LBPH works with grayscale)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                
                # Resize to standard size for consistency
                gray_resized = cv2.resize(gray, (160, 160))
                
                faces.append(gray_resized)
                labels.append(person_label)
                label_names.append(person_name)
                image_paths.append(str(img_path))
                img_count += 1
            
            except Exception as e:
                print(f"[WARN] Failed to process {img_path.name}: {e}")
        
        print(f"       ✓ Loaded {img_count} images for {person_name}")
    
    return faces, labels, label_names, image_paths, label_map

def train_lbph_model(faces, labels, label_names, label_map, val_split):
    """Train LBPH face recognizer"""
    n_samples = len(faces)
    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    
    print(f"\n{'='*60}")
    print(f"[INFO] LBPH FACE RECOGNITION")
    print(f"[INFO] Training with {n_samples} samples")
    print(f"[INFO] Number of people: {n_classes}")
    
    # Show label mapping
    print(f"\n[INFO] Label Mapping:")
    reverse_map = {v: k for k, v in label_map.items()}
    for label_id in sorted(reverse_map.keys()):
        name = reverse_map[label_id]
        count = labels.count(label_id)
        print(f"  {label_id}: {name} ({count} images)")
    
    # Split data for validation
    if val_split > 0 and n_samples > n_classes:
        indices = list(range(n_samples))
        random.shuffle(indices)
        split_idx = max(n_classes, int(n_samples * (1 - val_split)))
        
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
    else:
        train_idx = list(range(n_samples))
        val_idx = []
        print("\n[INFO] No validation - using all data for training")
    
    X_train = [faces[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    
    print(f"\n[INFO] Training samples: {len(train_idx)}")
    print(f"[INFO] Validation samples: {len(val_idx)}")
    print('='*60)
    
    # Create and train LBPH recognizer
    print(f"\n[INFO] Training LBPH Face Recognizer...")
    print(f"[INFO] Parameters:")
    print(f"  Radius: {args.radius}")
    print(f"  Neighbors: {args.neighbors}")
    print(f"  Grid X: {args.grid_x}")
    print(f"  Grid Y: {args.grid_y}")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=args.radius,
        neighbors=args.neighbors,
        grid_x=args.grid_x,
        grid_y=args.grid_y
    )
    
    recognizer.train(X_train, np.array(y_train))
    print("[INFO] Training complete!")
    
    # Validate
    if val_idx:
        print(f"\n{'='*60}")
        print(f"[VALIDATION RESULTS]")
        print(f"{'='*60}")
        
        correct = 0
        per_class_correct = Counter()
        per_class_total = Counter()
        
        for idx in val_idx:
            face = faces[idx]
            true_label = labels[idx]
            true_name = label_names[idx]
            
            # Predict
            pred_label, confidence = recognizer.predict(face)
            
            per_class_total[true_label] += 1
            
            if pred_label == true_label:
                correct += 1
                per_class_correct[true_label] += 1
        
        # Overall accuracy
        acc = correct / len(val_idx)
        print(f"Overall Accuracy: {acc:.3f} ({acc*100:.1f}%)")
        print(f"Correctly recognized: {correct}/{len(val_idx)} samples")
        
        # Per-class accuracy
        print("\nPer-person accuracy:")
        for label_id in sorted(per_class_total.keys()):
            name = reverse_map[label_id]
            class_acc = per_class_correct[label_id] / per_class_total[label_id]
            n_correct = per_class_correct[label_id]
            n_total = per_class_total[label_id]
            print(f"  {name}: {class_acc:.3f} ({class_acc*100:.1f}%) - {n_correct}/{n_total} correct")
        
        print('='*60)
    
    return recognizer

def main():
    random.seed(42)
    
    if not DATASET.exists():
        print(f"\n[ERROR] Dataset folder not found: {DATASET}")
        print("\n[HELP] To create dataset:")
        print("  1. Run: python 01_create_dataset_mediapipe.py")
        print("  2. Collect 50-100+ images per person")
        print("  3. Run this script again")
        raise SystemExit()
    
    MODELS.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    faces, labels, label_names, paths, label_map = load_dataset(DATASET)
    
    if len(faces) == 0:
        print("\n[ERROR] No images loaded!")
        print("\n[HELP] Troubleshooting:")
        print("  1. Check: dataset/person_name/*.jpg exists")
        print("  2. Verify images are readable")
        print("  3. Run dataset collection again")
        raise SystemExit()
    
    # Train LBPH model
    recognizer = train_lbph_model(faces, labels, label_names, label_map, args.val_split)
    
    # Save model and label mapping
    print(f"\n[INFO] Saving models...")
    
    # Save LBPH recognizer
    recognizer.write(str(MODEL_PKL))
    print(f"  ✓ LBPH model saved: {MODEL_PKL}")
    
    # Save label mapping (for converting predictions back to names)
    with open(LABEL_MAP_PKL, 'wb') as f:
        pickle.dump(label_map, f)
    print(f"  ✓ Label map saved: {LABEL_MAP_PKL}")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"Algorithm: LBPH (Local Binary Patterns Histograms)")
    print(f"Total samples: {len(faces)}")
    print(f"People: {', '.join(sorted(label_map.keys()))}")
    print("\n[NEXT STEPS]")
    print("  Webcam: python 04_predict_lbph_mediapipe.py")
    print("  Image:  python 04_predict_lbph_mediapipe.py --image photo.jpg")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()