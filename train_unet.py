import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("\n=== Training Baseline U-Net ===")

# -------------------------
# Unified Metrics
# -------------------------
def precision_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum(y_pred) - tp
    return tp / (tp + fp + 1e-7)

def recall_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true) - tp
    return tp / (tp + fn + 1e-7)

def f1_metric(y_true, y_pred):
    p = precision_metric(y_true, y_pred)
    r = recall_metric(y_true, y_pred)
    return 2*p*r/(p+r+1e-7)

def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter + 1e-7
    return inter/union

# -------------------------
# Load data
# -------------------------
def load_split(split):
    img_dir=f"./processed-loveda/{split}/images"
    mask_dir=f"./processed-loveda/{split}/masks"

    X=[np.squeeze(np.load(os.path.join(img_dir,f))) for f in sorted(os.listdir(img_dir))]
    y=[np.squeeze(np.load(os.path.join(mask_dir,f))) for f in sorted(os.listdir(mask_dir))]

    X=np.array(X,dtype=np.float32)/255.
    y=np.array([m[...,None] if m.ndim==2 else m for m in y],dtype=np.float32)
    return X,y

X_train, y_train=load_split("train")
X_val,   y_val  =load_split("val")
X_test,  y_test =load_split("test")

print("Train:",X_train.shape)
print("Val:",X_val.shape)
print("Test:",X_test.shape)

# -------------------------
# Simple U-Net
# -------------------------
def conv_block(x,filters):
    x=layers.Conv2D(filters,3,padding="same",activation="relu")(x)
    x=layers.Conv2D(filters,3,padding="same",activation="relu")(x)
    return x

def encoder_block(x,filters):
    f=conv_block(x,filters)
    p=layers.MaxPooling2D(2)(f)
    return f,p

def decoder_block(x,skip,filters):
    x=layers.Conv2DTranspose(filters,2,strides=2,padding="same")(x)
    x=layers.concatenate([x,skip])
    x=conv_block(x,filters)
    return x

def build_unet(input_shape):
    inputs=layers.Input(input_shape)
    s1,p1=encoder_block(inputs,16)
    s2,p2=encoder_block(p1,32)
    s3,p3=encoder_block(p2,64)
    s4,p4=encoder_block(p3,128)
    b=conv_block(p4,256)
    d1=decoder_block(b,s4,128)
    d2=decoder_block(d1,s3,64)
    d3=decoder_block(d2,s2,32)
    d4=decoder_block(d3,s1,16)
    outputs=layers.Conv2D(1,1,activation="sigmoid")(d4)
    return models.Model(inputs,outputs)

model=build_unet(X_train.shape[1:])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=[precision_metric, recall_metric, f1_metric, iou_metric]
)

history=model.fit(
    X_train,y_train,
    validation_data=(X_val,y_val),
    epochs=60,batch_size=2
)

model.save("unet_unified.h5")

print("\n=== Evaluation on Test Set ===")
results=model.evaluate(X_test,y_test)
print("Loss, Precision, Recall, F1, IoU =",results)

