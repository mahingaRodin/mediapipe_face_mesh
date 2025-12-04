#!/usr/bin/env python3
"""
Train Face Recognition Model - Single Person Support
Works with 1 or more persons using One-Class SVM for single person
"""
import argparse
import pickle
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATASET = ROOT / "dataset"
MODEL_PKL = MODELS / "face_recognition_model.pkl"
ENCODER_PKL = MODELS / "label_encoder.pkl"
EMBEDDINGS_PKL = MODELS / "face_embeddings.pkl"

# CLI
ap = argparse.ArgumentParser(description="Train face recognition model")
ap.add_argument("--val-split", type=float, default=0.2,
                help="Validation split (default: 0.2, set to 0 to skip validation)")
args = ap.parse_args()

def get_embedding(face_img):
    """Generate 128D embedding from face image"""
    # Resize to standard size
    face_resized = cv2.resize(face_img, (160, 160))
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) if len(face_resized.shape) == 3 else face_resized
    
    # Create embedding using histogram and spatial features
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    
    # Add spatial features (divide into grid)
    grid_size = 4
    h, w = gray.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    spatial_features = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            spatial_features.append(cell.mean())
            spatial_features.append(cell.std())
    
    # Combine features
    embedding = np.concatenate([hist, spatial_features])
    
    # Normalize to 128D
    if len(embedding) < 128:
        embedding = np.pad(embedding, (0, 128 - len(embedding)), 'constant')
    else:
        embedding = embedding[:128]
    
    # L2 normalize
    embedding = embedding / (np.linalg.norm(embedding) + 1e-7)
    
    return embedding

def load_dataset(dataset_path):
    """Load pre-cropped face images and generate embeddings"""
    embeddings = []
    labels = []
    image_paths = []
    
    print(f"\n[INFO] Loading dataset from {dataset_path}")
    
    # Iterate through person folders
    person_dirs = [d for d in sorted(dataset_path.iterdir()) 
                  if d.is_dir() and not d.name.startswith('.')]
    
    if not person_dirs:
        print(f"[ERROR] No person folders found in {dataset_path}")
        print("[INFO] Expected structure: dataset/person_name/*.jpg")
        return np.array([]), [], []
    
    for person_dir in person_dirs:
        person_name = person_dir.name
        print(f"[INFO] Processing {person_name}...")
        
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
                
                # Generate embedding
                embedding = get_embedding(img)
                
                embeddings.append(embedding)
                labels.append(person_name)
                image_paths.append(str(img_path))
                img_count += 1
            
            except Exception as e:
                print(f"[WARN] Failed to process {img_path.name}: {e}")
        
        print(f"       ✓ Loaded {img_count} images for {person_name}")
    
    return np.array(embeddings), labels, image_paths

def train_single_person_model(embeddings, person_name, val_split):
    """Train One-Class SVM for single person verification"""
    n_samples = len(embeddings)
    
    print(f"\n{'='*60}")
    print(f"[INFO] SINGLE PERSON MODE")
    print(f"[INFO] Training verification model for: {person_name}")
    print(f"[INFO] Total samples: {n_samples}")
    
    # Split data for validation
    if val_split > 0 and n_samples > 5:
        indices = list(range(n_samples))
        random.shuffle(indices)
        split_idx = int(n_samples * (1 - val_split))
        
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
    else:
        train_idx = list(range(n_samples))
        val_idx = []
        print("[INFO] No validation - using all data for training")
    
    X_train = embeddings[train_idx]
    X_val = embeddings[val_idx] if val_idx else None
    
    print(f"\n[INFO] Training samples: {len(train_idx)}")
    print(f"[INFO] Validation samples: {len(val_idx)}")
    print('='*60)
    
    # Train One-Class SVM (learns what is "normal" for this person)
    print("\n[INFO] Training One-Class SVM...")
    model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    model.fit(X_train)
    print("[INFO] Training complete!")
    
    # Create pseudo label encoder (single class)
    le = LabelEncoder()
    le.classes_ = np.array([person_name])
    
    # Validate
    if X_val is not None and len(X_val) > 0:
        predictions = model.predict(X_val)
        # OneClassSVM returns +1 for inliers (same person), -1 for outliers
        n_correct = (predictions == 1).sum()
        acc = n_correct / len(X_val)
        
        print(f"\n{'='*60}")
        print(f"[VALIDATION RESULTS]")
        print(f"{'='*60}")
        print(f"Verification Accuracy: {acc:.3f} ({acc*100:.1f}%)")
        print(f"Correctly verified: {n_correct}/{len(X_val)} samples")
        print('='*60)
    
    return model, le, 'single_person'

def train_multi_person_model(embeddings, labels, val_split):
    """Train multi-class SVM for multiple persons"""
    n_samples = len(embeddings)
    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    
    print(f"\n{'='*60}")
    print(f"[INFO] MULTI-PERSON MODE")
    print(f"[INFO] Training with {n_samples} samples")
    print(f"[INFO] Classes ({n_classes}): {unique_labels}")
    
    label_counts = Counter(labels)
    print("\n[INFO] Samples per person:")
    for name, count in sorted(label_counts.items()):
        print(f"  {name}: {count} images")
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    # Split data
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
    
    X_train = embeddings[train_idx]
    y_train = encoded_labels[train_idx]
    X_val = embeddings[val_idx] if val_idx else None
    y_val = encoded_labels[val_idx] if val_idx else None
    
    print(f"\n[INFO] Training samples: {len(train_idx)}")
    print(f"[INFO] Validation samples: {len(val_idx)}")
    print('='*60)
    
    # Train multi-class SVM
    print("\n[INFO] Training Multi-Class SVM...")
    model = SVC(kernel='linear', probability=True, C=1.0, gamma='scale')
    model.fit(X_train, y_train)
    print("[INFO] Training complete!")
    
    # Validate
    if X_val is not None and len(X_val) > 0:
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        print(f"\n{'='*60}")
        print(f"[VALIDATION RESULTS]")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {acc:.3f} ({acc*100:.1f}%)")
        
        # Per-class accuracy
        print("\nPer-class accuracy:")
        for i, name in enumerate(le.classes_):
            class_mask = (y_val == i)
            if class_mask.sum() > 0:
                class_acc = (y_pred[class_mask] == y_val[class_mask]).mean()
                n_val_samples = class_mask.sum()
                print(f"  {name}: {class_acc:.3f} ({class_acc*100:.1f}%) - {n_val_samples} sample(s)")
        
        # Confusion matrix
        if n_classes > 1:
            cm = confusion_matrix(y_val, y_pred)
            print("\nConfusion Matrix:")
            print(cm)
        print('='*60)
    
    return model, le, 'multi_person'

def main():
    random.seed(42)
    
    if not DATASET.exists():
        print(f"\n[ERROR] Dataset folder not found: {DATASET}")
        print("\n[HELP] To create dataset:")
        print("  1. Run: python 01_create_dataset_mediapipe.py")
        print("  2. Collect 50-100+ images")
        print("  3. Run this script again")
        raise SystemExit()
    
    MODELS.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    embeddings, labels, paths = load_dataset(DATASET)
    
    if len(embeddings) == 0:
        print("\n[ERROR] No images loaded!")
        print("\n[HELP] Troubleshooting:")
        print("  1. Check: dataset/person_name/*.jpg exists")
        print("  2. Verify images are readable")
        print("  3. Run dataset collection again")
        raise SystemExit()
    
    # Check if single or multi person
    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    
    if n_classes == 1:
        # Single person - use One-Class SVM
        model, label_encoder, mode = train_single_person_model(
            embeddings, unique_labels[0], args.val_split
        )
    else:
        # Multiple persons - use standard multi-class SVM
        model, label_encoder, mode = train_multi_person_model(
            embeddings, labels, args.val_split
        )
    
    # Save everything
    print(f"\n[INFO] Saving models...")
    
    # Save model with metadata
    model_data = {
        'model': model,
        'mode': mode
    }
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  ✓ Model saved: {MODEL_PKL}")
    
    with open(ENCODER_PKL, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  ✓ Encoder saved: {ENCODER_PKL}")
    
    data = {'embeddings': embeddings, 'labels': labels, 'paths': paths}
    with open(EMBEDDINGS_PKL, 'wb') as f:
        pickle.dump(data, f)
    print(f"  ✓ Embeddings saved: {EMBEDDINGS_PKL}")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"Mode: {mode.upper().replace('_', ' ')}")
    print(f"Total samples: {len(embeddings)}")
    print(f"Classes: {', '.join(label_encoder.classes_)}")
    print("\n[NEXT STEPS]")
    print("  Webcam: python 04_predict_mediapipe.py")
    print("  Image:  python 04_predict_mediapipe.py --image photo.jpg")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()