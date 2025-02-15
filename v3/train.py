import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# PARAMETERS
NUM_FRAMES = 16         # fixed number of frames per video
IMG_SIZE = 224          # height and width
BATCH_SIZE = 256        # global batch size: with 8 TPU cores, each gets 32
EPOCHS = 50
NUM_CLASSES = 1         # binary classification; output sigmoid

# -------------------------------
# DATA GENERATOR FUNCTIONALITY
# -------------------------------
# Updated _load_video includes 40-sec clip normalization.

def _load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
    orig_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames_allowed = int(fps * 40)
    total_frames = min(orig_total_frames, max_frames_allowed)

    if total_frames < NUM_FRAMES:
        indices = list(range(total_frames))
    else:
        interval = total_frames / NUM_FRAMES
        indices = [int(i * interval) for i in range(NUM_FRAMES)]

    extracted = {}
    frame_id = 0
    ret = True
    while ret and frame_id < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in indices:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype('float32') / 255.0
            extracted[frame_id] = frame
        frame_id += 1
    cap.release()
    if not extracted:
        return np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    last_available = max(extracted.keys())
    out_frames = []
    for idx in indices:
        if idx in extracted:
            out_frames.append(extracted[idx])
        else:
            out_frames.append(extracted[last_available])
    return np.array(out_frames)

# Define a simple generator that yields (video, label) pairs.
def video_generator(video_paths, labels):
    for path, label in zip(video_paths, labels):
        yield _load_video(path), label

# Create a tf.data.Dataset from the generator.
def video_dataset(video_paths, labels, batch_size, shuffle=True):
    output_shape = ((NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3), ())
    dataset = tf.data.Dataset.from_generator(
        lambda: video_generator(video_paths, labels),
        output_types=(tf.float32, tf.float32),
        output_shapes=output_shape
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(video_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

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

video_paths, labels = load_video_paths_and_labels()
indices = np.arange(len(video_paths))
np.random.shuffle(indices)
split = int(0.8 * len(video_paths))
train_idx, val_idx = indices[:split], indices[split:]
train_paths = [video_paths[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]
val_paths = [video_paths[i] for i in val_idx]
val_labels = [labels[i] for i in val_idx]

train_ds = video_dataset(train_paths, train_labels, BATCH_SIZE, shuffle=True)
val_ds = video_dataset(val_paths, val_labels, BATCH_SIZE, shuffle=False)

# -------------------------------
# TPU INITIALIZATION & MODEL DEFINITION
# -------------------------------
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local') # Detect TPU    strategy = tf.distribute.TPUStrategy(tpu)
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except tf.errors.NotFoundError:
    strategy = tf.distribute.MirroredStrategy()

def compute_edge_magnitude(x):
    # x has shape (batch, H, W, 3)
    sobel = tf.image.sobel_edges(x)  # shape becomes (batch, H, W, 3, 2)
    # Compute the magnitude for each color channel:
    mag = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))  # now shape: (batch, H, W, 3)
    # Combine across the RGB channels (for example, take the mean):
    mag = tf.reduce_mean(mag, axis=-1, keepdims=True)  # final shape: (batch, H, W, 1)
    return mag


with strategy.scope():
    video_input = keras.Input(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    # edge_layer = layers.TimeDistributed(layers.Lambda(lambda x: tf.image.sobel_edges(x)))(video_input)
    edge_layer = layers.TimeDistributed(layers.Lambda(compute_edge_magnitude))(video_input)

    def frame_feature_extractor():
        inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
        x = layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.25)(x)
        x = layers.GlobalAveragePooling2D()(x)
        return keras.Model(inputs=inp, outputs=x, name="frame_feature_extractor")

    frame_extractor = frame_feature_extractor()
    frame_features = layers.TimeDistributed(frame_extractor)(edge_layer)

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
# CALLBACKS
# -------------------------------
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint('video_classifier_model_2.h5', monitor='val_loss', save_best_only=True)

# -------------------------------
# TRAINING
# -------------------------------
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                    callbacks=[early_stop, checkpoint])

model.save('final_video_classifier_model_2.h5')
