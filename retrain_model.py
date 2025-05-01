#!/usr/bin/env python3
import os, sys, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = 'backend/data'        # JSON user corrections
H5_MODEL    = 'model.h5'            # your base or last checkpoint
SM_DIR      = 'model_saved'         # SavedModel dir
TFJS_DIR    = 'public/tfjs_model'   # TFJS graph output
MIN_SAMPLES = 100                   # skip val_split under this
PRE_EPOCHS  = 3                     # MNIST pretrain
FT_EPOCHS   = 5                     # fine-tune
BATCH       = 64
# ────────────────────────────────────────────────────────────────────────────────

# 1) Load MNIST
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train[...,None]/255.0
y_train = to_categorical(y_train,10)
print(f"📦 MNIST: {len(x_train)} samples")

# 2) Load user corrections
ux, uy = [], []
for fn in os.listdir(DATA_DIR):
    if fn.endswith('.json'):
        o = json.load(open(os.path.join(DATA_DIR,fn)))
        ux.append(o['grayData'])
        uy.append(o['label'])
if ux:
    ux = np.array(ux,dtype='float32').reshape(-1,28,28,1)
    uy = to_categorical(uy,10)
    X = np.vstack([x_train, ux])
    Y = np.vstack([y_train, uy])
    print(f"🔗 +{len(ux)} user samples → total {len(X)}")
else:
    X, Y = x_train, y_train
    print("ℹ️ No user data; using MNIST only")

# 3) Load or build model
if os.path.exists(H5_MODEL):
    print("🔄 Loading", H5_MODEL)
    model = tf.keras.models.load_model(H5_MODEL)
else:
    print("🆕 Building new model")
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax'),
    ])

model.compile('adam','categorical_crossentropy',['accuracy'])

# 4) Pretrain on MNIST if brand new
if not os.path.exists(H5_MODEL):
    print(f"🚀 Pretraining ({PRE_EPOCHS} epochs)")
    model.fit(x_train,y_train,epochs=PRE_EPOCHS,
              batch_size=BATCH,validation_split=0.1,shuffle=True)

# 5) Fine-tune on combined
print(f"🚀 Fine-tuning ({FT_EPOCHS} epochs)")
fit_args = dict(epochs=FT_EPOCHS,batch_size=BATCH,shuffle=True)
if len(X)>=MIN_SAMPLES: fit_args['validation_split']=0.1
model.fit(X,Y,**fit_args)

# 6) Save H5
print("💾 Saving H5:", H5_MODEL)
model.save(H5_MODEL)

# 7) Export SavedModel
print("💾 Export SavedModel:", SM_DIR)
if os.path.isdir(SM_DIR): tf.io.gfile.rmtree(SM_DIR)
model.export(SM_DIR)

# 8) Convert to TFJS
print("🔧 Converting to TFJS:", TFJS_DIR)
if os.path.isdir(TFJS_DIR): tf.io.gfile.rmtree(TFJS_DIR)
os.system(
  "tensorflowjs_converter "
  "--input_format=tf_saved_model --output_format=tfjs_graph_model "
  f"{SM_DIR} {TFJS_DIR}"
)

print("✅ Retraining & conversion complete!")