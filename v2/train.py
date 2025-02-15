import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import Sequence

# PARAMETERS
NUM_FRAMES = 16         # fixed number of frames per video
IMG_SIZE = 224          # height and width of each frame
BATCH_SIZE = 64         # increased for TPU (must be divisible by 8)
EPOCHS = 50
NUM_CLASSES = 1         # binary classification; output sigmoid

# -------------------------------
# DATA GENERATOR FUNCTIONALITY
# -------------------------------
# (Keep your VideoDataGenerator _load_video implementation which now includes 40-sec normalization)

def _load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")

    # Get frames per second; if unavailable, assume 30 fps.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    # Limit video duration to 40 seconds
    orig_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames_allowed = int(fps * 40)
    total_frames = min(orig_total_frames, max_frames_allowed)

    # Determine sampling indices uniformly across the available frames.
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
            # Resize, convert BGR->RGB, and normalize.
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype('float32') / 255.0
            extracted[frame_id] = frame
        frame_id += 1
    cap.release()
    # If not enough frames, pad using the last available frame.
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

# Instead of using a Keras Sequence subclass, we create a generator function
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

# Load video file paths and labels from your directories.
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
# MODEL DEFINITION (inside TPU strategy scope)
# -------------------------------
# TPU initialization and TPUStrategy creation (see TensorFlow TPU guide :contentReference[oaicite:0]{index=0})
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # leave empty for Colab TPU
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    # Input shape: (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    video_input = keras.Input(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3))

    # Apply edge extraction per frame.
    edge_layer = layers.TimeDistributed(layers.Lambda(lambda x: tf.image.sobel_edges(x)))(video_input)

    # Define a frame-level CNN feature extractor.
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
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        return keras.Model(inputs=inp, outputs=x, name="frame_feature_extractor")

    frame_extractor = frame_feature_extractor()
    # Apply the frame extractor to each frame (note: edge_layer has 1 channel now)
    frame_features = layers.TimeDistributed(frame_extractor)(edge_layer)

    # Temporal aggregation with an LSTM.
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
checkpoint = callbacks.ModelCheckpoint('video_classifier_model.h5', monitor='val_loss', save_best_only=True)

# -------------------------------
# TRAINING
# -------------------------------
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                    callbacks=[early_stop, checkpoint])

# Save the final model.
model.save('./v3/final_video_classifier_model.h5')
# Save the training history.
np.save('./v3/training_history.npy', history.history)
# Save the model architecture.
with open('./v3/model_architecture.json', 'w') as f:
    f.write(model.to_json())
# Save the model weights.
model.save_weights('./v3/model_weights.h5')
# Save the model configuration.
with open('./v3/model_config.json', 'w') as f:
    json.dump(model.get_config(), f)
# Save the model optimizer state.
with open('./v3/optimizer_state.pkl', 'wb') as f:
    pickle.dump(model.optimizer.get_weights(), f)
# Save the model training history.
with open('./v3/training_history.json', 'w') as f:
    json.dump(history.history, f)
# Save the model training configuration.
with open('./v3/training_config.json', 'w') as f:
    json.dump(model.optimizer.get_config(), f)
# Save the model training parameters.
with open('./v3/training_params.json', 'w') as f:
    json.dump({'batch_size': BATCH_SIZE, 'epochs': EPOCHS}, f)