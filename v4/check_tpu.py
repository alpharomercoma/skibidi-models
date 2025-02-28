import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import Sequence

# -------------------------------
# TPU SETUP and MIXED PRECISION
# -------------------------------
cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)

# Enable mixed precision for TPUs (bfloat16 is optimal on TPU hardware)
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# -------------------------------
# PARAMETERS (Adjusted for TPU usage)
# -------------------------------
NUM_FRAMES = 16       # fixed number of frames per video clip
IMG_SIZE = 224        # height and width of frames
BATCH_SIZE = 16       # increased batch size to leverage TPU parallelism
EPOCHS = 50
NUM_CLASSES = 1       # binary classification output with sigmoid activation

# -------------------------------
# DATA PREPARATION FUNCTIONS
# -------------------------------
def load_video_frames(path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames_allowed = int(fps * 35)
    total_frames = min(total_frames, max_frames_allowed)

    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        interval = total_frames / num_frames
        indices = [int(i * interval) for i in range(num_frames)]

    extracted = {}
    frame_id = 0
    ret = True
    while ret and frame_id < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in indices:
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype('float32') / 255.0
            extracted[frame_id] = frame
        frame_id += 1
    cap.release()

    if not extracted:
        return np.zeros((num_frames, img_size, img_size, 3), dtype=np.float32)

    last = max(extracted.keys())
    out_frames = [extracted.get(idx, extracted[last]) for idx in indices]
    return np.array(out_frames)

def compute_optical_flow(frames):
    gray_frames = [cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for frame in frames]
    flow_frames = [np.zeros_like(gray_frames[0], dtype=np.float32)]
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
        diff = diff.astype('float32') / 255.0
        flow_frames.append(diff)
    flow_frames = [np.expand_dims(frame, axis=-1) for frame in flow_frames]
    return np.array(flow_frames)

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
        batch_indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        batch_video_paths = [self.video_paths[k] for k in batch_indexes]
        batch_labels = [self.labels[k] for k in batch_indexes]

        rgb_batch = np.zeros((len(batch_video_paths), self.num_frames, self.img_size, self.img_size, 3), dtype=np.float32)
        flow_batch = np.zeros((len(batch_video_paths), self.num_frames, self.img_size, self.img_size, 1), dtype=np.float32)

        for i, path in enumerate(batch_video_paths):
            frames = load_video_frames(path, self.num_frames, self.img_size)
            rgb_batch[i] = frames
            flow_batch[i] = compute_optical_flow(frames)

        y = np.array(batch_labels, dtype=np.float32)
        return ((rgb_batch, flow_batch), y)

def load_video_paths_and_labels():
    video_paths = []
    labels = []
    non_sludge_dir = './dataset/video/non_sludge'
    for fname in os.listdir(non_sludge_dir):
        if fname.lower().endswith(('.mp4', '.avi', '.mov')):
            video_paths.append(os.path.join(non_sludge_dir, fname))
            labels.append(0)
    sludge_dir = './dataset/video/sludge'
    for fname in os.listdir(sludge_dir):
        if fname.lower().endswith(('.mp4', '.avi', '.mov')):
            video_paths.append(os.path.join(sludge_dir, fname))
            labels.append(1)
    return video_paths, labels

# Split data into training and validation sets.
video_paths, labels = load_video_paths_and_labels()
indices = np.arange(len(video_paths))
np.random.shuffle(indices)
split = int(0.8 * len(video_paths))
train_idx, val_idx = indices[:split], indices[split:]
train_paths = [video_paths[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]
val_paths = [video_paths[i] for i in val_idx]
val_labels = [labels[i] for i in val_idx]

train_gen = VideoDataGenerator(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True)
val_gen = VideoDataGenerator(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# MODEL DEFINITION (within TPU strategy scope)
# -------------------------------
with strategy.scope():
    def build_3d_cnn_branch(input_shape, base_filters=32):
        inp = keras.Input(shape=input_shape)
        x = layers.Conv3D(base_filters, kernel_size=(3,3,3), padding='same', activation='relu')(inp)
        x = layers.MaxPooling3D(pool_size=(1,2,2))(x)
        x = layers.Conv3D(base_filters*2, kernel_size=(3,3,3), padding='same', activation='relu')(x)
        x = layers.MaxPooling3D(pool_size=(1,2,2))(x)
        x = layers.Conv3D(base_filters*4, kernel_size=(3,3,3), padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling3D()(x)
        return keras.Model(inputs=inp, outputs=x)

    # Build branch for raw RGB input
    rgb_input = keras.Input(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    rgb_branch = build_3d_cnn_branch((NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3), base_filters=32)
    rgb_features = rgb_branch(rgb_input)

    # Build branch for optical flow input
    flow_input = keras.Input(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 1))
    flow_branch = build_3d_cnn_branch((NUM_FRAMES, IMG_SIZE, IMG_SIZE, 1), base_filters=16)
    flow_features = flow_branch(flow_input)

    # Fuse features and add Dense layers.
    fused = layers.Concatenate()([rgb_features, flow_features])
    x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(fused)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(NUM_CLASSES, activation='sigmoid', dtype='float32')(x)  # Cast output to float32 for stability

    model = keras.Model(inputs=[rgb_input, flow_input], outputs=output)
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

# Save the final model (best weights saved via checkpoint)
model.save('final_video_classifier_model.h5')
