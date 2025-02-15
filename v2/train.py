import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import Sequence

# PARAMETERS
NUM_FRAMES = 16         # fixed number of frames per video
IMG_SIZE = 224          # height and width
BATCH_SIZE = 8          # adjust based on your GPU RAM
EPOCHS = 50
NUM_CLASSES = 1         # binary classification; output sigmoid

# DATA GENERATOR FOR VIDEO FILES
class VideoDataGenerator(Sequence):
    def __init__(self, video_paths, labels, batch_size=BATCH_SIZE, num_frames=NUM_FRAMES, img_size=IMG_SIZE, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.video_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        batch_video_paths = [self.video_paths[k] for k in batch_indexes]
        batch_labels = [self.labels[k] for k in batch_indexes]

        # Process each video
        X = np.zeros((len(batch_video_paths), self.num_frames, self.img_size, self.img_size, 3), dtype=np.float32)
        for i, path in enumerate(batch_video_paths):
            frames = self._load_video(path)
            X[i] = frames
        y = np.array(batch_labels, dtype=np.float32)
        return X, y

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")

        # Get frames per second; if unavailable, assume 30 fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0

        # Original total frames and the maximum allowed (40 seconds)
        orig_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames_allowed = int(fps * 10)
        # Use the shorter between the video’s total frames and 40 seconds worth of frames.
        total_frames = min(orig_total_frames, max_frames_allowed)

        # Compute frame sampling indices uniformly over the available segment.
        if total_frames < self.num_frames:
            indices = list(range(total_frames))
        else:
            interval = total_frames / self.num_frames
            indices = [int(i * interval) for i in range(self.num_frames)]

        # Read frames sequentially up to total_frames.
        extracted = {}
        frame_id = 0
        ret = True
        while ret and frame_id < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id in indices:
                # Resize frame and convert from BGR to RGB, normalize pixel values.
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype('float32') / 255.0
                extracted[frame_id] = frame
            frame_id += 1
        cap.release()

        # If no frames were extracted, return an array of zeros.
        if not extracted:
            return np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.float32)

        # For any index not found (if video is shorter than self.num_frames), pad with last available frame.
        last_available = max(extracted.keys())
        out_frames = []
        for idx in indices:
            if idx in extracted:
                out_frames.append(extracted[idx])
            else:
                out_frames.append(extracted[last_available])

        return np.array(out_frames)

# Gather video file paths and labels from directories
def load_video_paths_and_labels():
    video_paths = []
    labels = []
    # non_sludge: label 0
    non_sludge_dir = './dataset/video/non_sludge'
    for fname in os.listdir(non_sludge_dir):
        if fname.lower().endswith(('.mp4', '.avi', '.mov')):
            video_paths.append(os.path.join(non_sludge_dir, fname))
            labels.append(0)
    # sludge: label 1
    sludge_dir = './dataset/video/sludge'
    for fname in os.listdir(sludge_dir):
        if fname.lower().endswith(('.mp4', '.avi', '.mov')):
            video_paths.append(os.path.join(sludge_dir, fname))
            labels.append(1)
    return video_paths, labels

# Split into training and validation sets (e.g., 80/20)
video_paths, labels = load_video_paths_and_labels()
indices = np.arange(len(video_paths))
np.random.shuffle(indices)
split = int(0.8 * len(video_paths))
train_idx, val_idx = indices[:split], indices[split:]
train_paths = [video_paths[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]
val_paths = [video_paths[i] for i in val_idx]
val_labels = [labels[i] for i in val_idx]

train_gen = VideoDataGenerator(train_paths, train_labels, batch_size=BATCH_SIZE)
val_gen = VideoDataGenerator(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# MODEL DEFINITION
# -------------------------------

# 1. Edge extraction layer: using sobel_edges to compute gradient magnitude.
def edge_extraction(x):
    # x shape: (H, W, 3)
    sobel = tf.image.sobel_edges(x)  # shape: (H, W, 3, 2)
    # Compute gradient magnitude per channel
    grad_mag = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))  # shape: (H, W, 3)
    # Average over channels to get one channel edge map
    grad_edge = tf.reduce_mean(grad_mag, axis=-1, keepdims=True)  # (H, W, 1)
    # Normalize to [0,1]
    max_val = tf.reduce_max(grad_edge) + 1e-6
    grad_edge = grad_edge / max_val
    return grad_edge

# 2. Spatial attention module (applied per frame)
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        # Create the Conv2D layer here (only once)
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal'
        )

    def call(self, inputs):
        # Compute average and max pooling across channels
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention


# Build the model: Input shape = (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
video_input = keras.Input(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3))

# Apply edge extraction to each frame
edge_layer = layers.TimeDistributed(layers.Lambda(edge_extraction))(video_input)
# Now edge_layer shape: (batch, NUM_FRAMES, IMG_SIZE, IMG_SIZE, 1)

# TimeDistributed CNN feature extractor
def frame_feature_extractor():
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    # First conv block
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)
    # Second conv block
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)
    # Apply spatial attention
    # x = layers.Lambda(spatial_attention_module)(x)
    x = SpatialAttention(kernel_size=7)(x)
    # Global average pooling per frame
    x = layers.GlobalAveragePooling2D()(x)
    model = keras.Model(inputs=inp, outputs=x, name="frame_feature_extractor")
    return model

frame_extractor = frame_feature_extractor()
# Apply to each frame
frame_features = layers.TimeDistributed(frame_extractor)(edge_layer)
# frame_features shape: (batch, NUM_FRAMES, feature_dim)

# Aggregate temporal information using an LSTM
x = layers.LSTM(64, return_sequences=False)(frame_features)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(x)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
output = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)

model = keras.Model(inputs=video_input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# CALLBACKS: EarlyStopping and ModelCheckpoint
# -------------------------------
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint('video_classifier_model.h5', monitor='val_loss', save_best_only=True)

# -------------------------------
# TRAINING
# -------------------------------
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[early_stop, checkpoint])

# Save the final model (best weights were saved by the checkpoint callback)
model.save('final_video_classifier_model.h5')
