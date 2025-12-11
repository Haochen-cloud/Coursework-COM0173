import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

print("\n=== Transfer Learning Attention U-Net ===")

# -------------------------
# Load dataset
# -------------------------
def load_split(split):
    img_dir = f"./processed-loveda/{split}/images"
    mask_dir = f"./processed-loveda/{split}/masks"

    X = [np.squeeze(np.load(os.path.join(img_dir,f))) for f in sorted(os.listdir(img_dir))]
    y = [np.squeeze(np.load(os.path.join(mask_dir,f))) for f in sorted(os.listdir(mask_dir))]

    X = np.array(X, dtype=np.float32) / 255.
    y = np.array([m[...,None] if m.ndim==2 else m for m in y], dtype=np.float32)
    return X, y


X_train, y_train = load_split("train")
X_val,   y_val   = load_split("val")
X_test,  y_test  = load_split("test")

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# -------------------------
# Load pretrained model
# -------------------------
print("\nLoading pretrained Attention U-Net...")
model = load_model("./unet-attention-3d.hdf5", compile=False)

# Freeze backbone
for layer in model.layers[:56]:
    layer.trainable = False


# compile with ORIGINAL LOSS
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy"
)

model.summary()


# -------------------------
# Training hyper-parameter
# -------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=1,
    shuffle=True
)

model.save("att_unet_transfer_safe.h5")

# -------------------------
# External metrics
# -------------------------
def evaluate_external(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(np.float32)

    tp = np.sum(y_pred_bin * y_true)
    fp = np.sum(y_pred_bin) - tp
    fn = np.sum(y_true) - tp

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * precision * recall / (precision + recall + 1e-7)

    inter = np.sum(y_pred_bin * y_true)
    union = np.sum(y_pred_bin) + np.sum(y_true) - inter + 1e-7
    iou = inter / union

    return precision, recall, f1, iou


print("\n=== Evaluating on Test Set ===")
y_pred = model.predict(X_test, batch_size=1)

precision, recall, f1, iou = evaluate_external(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"IoU:       {iou:.4f}")

