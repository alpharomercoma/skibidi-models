import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------
#  MODEL LOADING AND INFERENCE
# -------------------------------

# --- Constants (Match training script) ---
NUM_FRAMES = 16
IMG_SIZE = 224
NUM_CLASSES = 1  # Binary classification

# --- TPU Setup (Optional, for inference if model was TPU trained) ---
# If you are NOT using a TPU for inference, comment out the TPU setup section.
cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)

# --- Data Preparation Functions (MUST MATCH TRAINING) ---
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

    if not extracted:  # Handle empty videos
        return np.zeros((num_frames, img_size, img_size, 3), dtype=np.float32)

    last = max(extracted.keys())
    out_frames = [extracted.get(idx, extracted[last]) for idx in indices]
    return np.array(out_frames)

def compute_optical_flow(frames):
    gray_frames = [cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for frame in frames]
    flow_frames = [np.zeros_like(gray_frames[0], dtype=np.float32)]  # Initialize with a zero frame
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
        diff = diff.astype('float32') / 255.0
        flow_frames.append(diff)
    flow_frames = [np.expand_dims(frame, axis=-1) for frame in flow_frames]  # Add channel dimension
    return np.array(flow_frames)

# --- Model Building Function (MUST MATCH TRAINING) ---
def build_3d_cnn_branch(input_shape, base_filters=32):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv3D(base_filters, kernel_size=(3,3,3), padding='same', activation='relu')(inp)
    x = layers.MaxPooling3D(pool_size=(1,2,2))(x)
    x = layers.Conv3D(base_filters*2, kernel_size=(3,3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling3D(pool_size=(1,2,2))(x)
    x = layers.Conv3D(base_filters*4, kernel_size=(3,3,3), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling3D()(x)
    return keras.Model(inputs=inp, outputs=x)

def build_model():
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
    output = layers.Dense(NUM_CLASSES, activation='sigmoid', dtype='float32')(x)

    model = keras.Model(inputs=[rgb_input, flow_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #must compile, even for inference
    return model

# --- Load the Model ---
# Use strategy.scope() if loading a model trained on TPU.
with strategy.scope():  # Remove this line if not using TPU for inference
      model = build_model()
      model.load_weights('final_video_classifier_model.h5')  # Or the path to your best saved model


# --- Prediction Function ---
def classify_video(video_path, model):
    """Classifies a video as sludge (1) or non-sludge (0).

    Args:
        video_path: Path to the video file.
        model: The loaded Keras model.

    Returns:
        A tuple: (prediction, confidence).  Prediction is 0 or 1,
        confidence is the model's output (probability).
    """
    frames = load_video_frames(video_path)
    flow = compute_optical_flow(frames)

    # Add batch dimension (required by the model)
    frames = np.expand_dims(frames, axis=0)
    flow = np.expand_dims(flow, axis=0)

    # Get the model's prediction (probability of being sludge)
    prediction_probability = model.predict([frames, flow])[0][0]

    # Classify based on probability
    prediction = 1 if prediction_probability >= 0.5 else 0
    confidence = prediction_probability

    return prediction, confidence

# --- Example Usage ---
video_to_classify = '/home/alpha/alpha/skibidi-models/manual/sludge_4.mp4'  # Replace with your video
#video_to_classify = 'path/to/your/video.mp4'
prediction, confidence = classify_video(video_to_classify, model)


if prediction == 1:
    print(f"The video is classified as SLUDGE with confidence: {confidence:.4f}")
else:
    print(f"The video is classified as NON-SLUDGE with confidence: {1-confidence:.4f}")