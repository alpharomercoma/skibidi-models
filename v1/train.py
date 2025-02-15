import os
import glob
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------
# 1. Data pipeline and video loading
# -------------------------------

# Constants
NUM_FRAMES = 16    # number of frames to sample per video
IMG_SIZE = 224     # target spatial resolution
BATCH_SIZE = 8
EPOCHS = 50

def load_video(path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    """
    Loads a video from disk (using OpenCV), samples 'num_frames' uniformly,
    resizes each frame to (img_size, img_size) and normalizes pixel values.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = num_frames
    # Uniformly sample frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        frame_idx += 1
    cap.release()
    # Pad if not enough frames
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])
    return np.array(frames)  # shape: (num_frames, img_size, img_size, 3)

def load_video_tf(path, label):
    # Wrap the video-loading function so it can be used in tf.data
    video = tf.py_function(func=lambda p: load_video(p.numpy().decode('utf-8')),
                           inp=[path],
                           Tout=tf.float32)
    video.set_shape((NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    return video, label

def create_dataset(data_dir):
    """
    Expects two folders under data_dir:
      - sludge (videos labeled 1)
      - non_sludge (videos labeled 0)
    """
    sludge_paths = glob.glob(os.path.join(data_dir, "sludge", "*.mp4"))
    non_sludge_paths = glob.glob(os.path.join(data_dir, "non_sludge", "*.mp4"))
    all_paths = sludge_paths + non_sludge_paths
    labels = [1] * len(sludge_paths) + [0] * len(non_sludge_paths)
    combined = list(zip(all_paths, labels))
    random.shuffle(combined)
    all_paths, labels = zip(*combined)
    return list(all_paths), list(labels)

# Adjust the following DATA_DIR if needed.
DATA_DIR = "./dataset/video"  # Assumes /dataset/sludge and /dataset/non_sludge exist.
file_paths, file_labels = create_dataset(DATA_DIR)

# Split dataset into training and testing (80/20 split)
split_idx = int(0.8 * len(file_paths))
train_paths, train_labels = file_paths[:split_idx], file_labels[:split_idx]
test_paths, test_labels = file_paths[split_idx:], file_labels[split_idx:]

# Create tf.data.Datasets for training and testing.
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(load_video_tf, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.map(load_video_tf, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# 2. Model: Spatial attention + CNN backbone + temporal pooling
# -------------------------------

# Spatial Attention module (inspired by CBAM)
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters=1,
                                  kernel_size=kernel_size,
                                  padding='same',
                                  activation='sigmoid',
                                  kernel_initializer='he_normal')
    def call(self, inputs):
        # inputs shape: (batch, H, W, C)
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

def build_model(num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    # Input: video clip of shape (num_frames, img_size, img_size, 3)
    video_input = layers.Input(shape=(num_frames, img_size, img_size, 3))

    # Use a pre-trained ResNet50 as the frame-level feature extractor.
    # We do NOT apply global pooling here in order to retain spatial layout.
    base_cnn = tf.keras.applications.ResNet50(include_top=False,
                                              weights='imagenet',
                                              input_shape=(img_size, img_size, 3))
    base_cnn.trainable = False  # freeze backbone

    # Process each frame with the CNN backbone via TimeDistributed.
    # Expected output per frame: feature map of shape (7, 7, 2048)
    td_features = layers.TimeDistributed(base_cnn)(video_input)

    # Apply spatial attention on each frame's feature map.
    td_attended = layers.TimeDistributed(SpatialAttention())(td_features)

    # For each frame, perform global average pooling to obtain a feature vector.
    td_pooled = layers.TimeDistributed(layers.GlobalAveragePooling2D())(td_attended)
    # Now shape is (batch, num_frames, feature_dim)

    # Aggregate the temporal dimension via average pooling.
    video_features = layers.GlobalAveragePooling1D()(td_pooled)

    # Classification head.
    x = layers.Dense(256, activation='relu')(video_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)  # binary classification

    model = keras.Model(inputs=video_input, outputs=output)
    return model

# -------------------------------
# 3. TPU Strategy and Model Compilation
# -------------------------------

try:
    # Detect TPU and initialize the TPU system.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except Exception as e:
    print('TPU not found, using default strategy')
    strategy = tf.distribute.get_strategy()

with strategy.scope():
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

# -------------------------------
# 4. Training, Evaluation, and Saving
# -------------------------------

# Callbacks: early stopping to prevent overfitting.
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

history = model.fit(train_ds,
                    validation_data=test_ds,
                    epochs=EPOCHS,
                    callbacks=callbacks)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Save the trained model.
model.save('video_classification_sludge_model.h5')
