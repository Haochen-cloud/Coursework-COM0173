import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_iou(pred, true):
    intersection = np.logical_and(true, pred).sum()
    union = np.logical_or(true, pred).sum()
    return intersection / union if union > 0 else 0


# ============================================
# Utility: load dataset (test only)
# ============================================

def load_dataset_from_folder(base_path):
    img_dir = os.path.join(base_path, "test", "images")
    mask_dir = os.path.join(base_path, "test", "masks")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])

    X_list, y_list = [], []

    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        if not os.path.exists(mask_path):
            print("Warning: Missing mask for", fname)
            continue

        img = np.load(img_path)
        mask = np.load(mask_path)

        # remove leading dimension if exists (img)
        # (1,512,512,4) → (512,512,4)
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]

        # remove leading dimension if exists (mask)
        # cases:
        # (1,512,512) → (512,512),(1,512,512,1) → (512,512,1)
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask[0]
        elif mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]


        if mask.ndim == 2:
            mask = mask.reshape(512, 512, 1)

        X_list.append(img)
        y_list.append(mask)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Loaded {len(X)} samples from {base_path}")
    print("X shape =", X.shape)
    print("y shape =", y.shape)
    return X, y



# ============================================
# Utility: evaluate model
# ============================================

def evaluate_model(model, X, y):
    """
    X: (N, 512, 512, 4)
    y: (N, 512, 512, 1)
    """
    ious, precisions, recalls, f1s = [], [], [], []

    n = len(X)
    print(f"Evaluating on {n} samples (batch size = 1)...")

    for i in range(n):
        x_i = X[i:i+1]
        y_i = y[i:i+1]

        pred = model.predict(x_i, verbose=0)   # (1, 512, 512, 1)
        pred_bin = (pred > 0.5).astype("uint8")

        p = pred_bin.reshape(-1)
        t = y_i.reshape(-1)

        iou = compute_iou(p, t)
        prec = precision_score(t, p, zero_division=0)
        rec = recall_score(t, p, zero_division=0)
        f1 = f1_score(t, p, zero_division=0)

        ious.append(iou)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        if (i+1) % 5 == 0 or (i+1) == n:
            print(f"  processed {i+1}/{n} patches")

    return {
        "IoU": float(np.mean(ious)),
        "Precision": float(np.mean(precisions)),
        "Recall": float(np.mean(recalls)),
        "F1-score": float(np.mean(f1s))
    }



def main():

    print("Loading 4-band Attention U-Net models...")
    model_amazon = load_model("unet-attention-4d.hdf5")
    model_atlantic = load_model("unet-attention-4d-atlantic.hdf5")

    # -----------------------------------------
    # Evaluate 4-band Amazon
    # -----------------------------------------
    print("\n=== Evaluating 4-band Amazon test set ===")
    X_a, y_a = load_dataset_from_folder("./amazon-processed-large")
    result_amazon = evaluate_model(model_amazon, X_a, y_a)

    print("\n4-band Amazon Results:")
    for k, v in result_amazon.items():
        print(f"{k}: {v:.4f}")

    # -----------------------------------------
    # Evaluate 4-band Atlantic Forest
    # -----------------------------------------
    print("\n=== Evaluating 4-band Atlantic test set ===")
    X_af, y_af = load_dataset_from_folder("./atlantic-processed-large")
    result_atlantic = evaluate_model(model_atlantic, X_af, y_af)

    print("\n4-band Atlantic Results:")
    for k, v in result_atlantic.items():
        print(f"{k}: {v:.4f}")

    print("\n=== DONE. These are your Part A-1 reproduction metrics. ===")


if __name__ == "__main__":
    main()
