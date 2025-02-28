import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse

# -------------------------------
# CUSTOM CAST LAYER DEFINITION
# -------------------------------
@tf.keras.utils.register_keras_serializable()
class Cast(tf.keras.layers.Layer):
    def __init__(self, target_dtype=None, **kwargs):
        # Allow configuration via 'dtype' if target_dtype not explicitly provided.
        if target_dtype is None and 'dtype' in kwargs:
            target_dtype = kwargs.pop('dtype')
        if target_dtype is None:
            raise ValueError("target_dtype must be specified for Cast layer.")
        super(Cast, self).__init__(**kwargs)
        self.target_dtype = target_dtype

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"target_dtype": self.target_dtype})
        return config

# -------------------------------
# PARAMETERS
# -------------------------------
NUM_FRAMES = 16       # fixed number of frames expected by the model
IMG_SIZE = 224        # height and width to which frames are resized

# -------------------------------
# DATA PREPARATION FUNCTIONS
# -------------------------------
def load_video_frames(path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    """
    Loads the entire video (up to 35 seconds) and resizes frames.
    If the video is shorter than the required number of frames,
    the last frame is repeated (extended) until num_frames frames are returned.
    If the video is longer, frames are uniformly sampled.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames_allowed = int(fps * 35)  # process maximum of 35 seconds
    total_frames = min(total_frames, max_frames_allowed)

    frames = []
    frame_id = 0
    ret = True
    while ret and frame_id < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize and convert to RGB
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype('float32') / 255.0
        frames.append(frame)
        frame_id += 1
    cap.release()

    if len(frames) == 0:
        return np.zeros((num_frames, img_size, img_size, 3), dtype=np.float32)

    if len(frames) < num_frames:
        # Extend by repeating the last frame until num_frames is reached
        last_frame = frames[-1]
        while len(frames) < num_frames:
            frames.append(last_frame)
        return np.array(frames)
    else:
        # Uniformly sample frames from the entire video duration
        indices = np.linspace(0, len(frames) - 1, num=num_frames, dtype=int)
        sampled_frames = [frames[idx] for idx in indices]
        return np.array(sampled_frames)

def compute_optical_flow(frames):
    """
    Computes a simple frame-to-frame absolute difference as a proxy for optical flow.
    Returns a sequence of single-channel flow images.
    """
    gray_frames = [cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                   for frame in frames]
    flow_frames = [np.zeros_like(gray_frames[0], dtype=np.float32)]
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
        diff = diff.astype('float32') / 255.0
        flow_frames.append(diff)
    flow_frames = [np.expand_dims(frame, axis=-1) for frame in flow_frames]
    return np.array(flow_frames)

def compute_edges(frames, img_size=IMG_SIZE):
    """
    Computes Canny edges on each frame after applying a Gaussian blur.
    Every detail is preserved by processing each frame individually.
    Returns a sequence of edge maps (single channel).
    """
    edge_frames = []
    for frame in frames:
        # Convert to grayscale (using full dynamic range)
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # Apply Gaussian blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Perform Canny edge detection with tuned thresholds
        edges = cv2.Canny(blurred, 50, 150)
        # Normalize edge map to [0,1] and add channel dimension
        edges = edges.astype('float32') / 255.0
        edges = np.expand_dims(edges, axis=-1)
        edge_frames.append(edges)
    return np.array(edge_frames)

# -------------------------------
# LOAD THE SAVED MODEL ON CPU
# -------------------------------
try:
    # Force model loading under a CPU strategy to avoid TPU transfer issues
    strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
    with strategy.scope():
        with tf.keras.utils.custom_object_scope({'Cast': Cast}):
            model = keras.models.load_model('final_video_classifier_model.h5')
    print("Model loaded successfully on CPU.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# -------------------------------
# CLASSIFICATION FUNCTION
# -------------------------------
def classify_video(video_path):
    """
    Processes the video (entire video up to 35 seconds) and computes:
      - Raw RGB frames (sampled or padded to NUM_FRAMES)
      - Optical flow between frames
      - Detailed edge maps from each frame
    Runs inference using the loaded model to predict sludge (1) vs non-sludge (0)
    and returns both the predicted label and the associated confidence.
    """
    # Load video frames
    frames = load_video_frames(video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE)

    # Prepare model inputs by computing supplementary channels
    rgb_input = np.expand_dims(frames, axis=0)
    flow_input = np.expand_dims(compute_optical_flow(frames), axis=0)
    edge_input = np.expand_dims(compute_edges(frames, img_size=IMG_SIZE), axis=0)

    # Run inference (model was loaded under a CPU strategy)
    pred = model.predict((rgb_input, flow_input, edge_input))[0][0]
    pred2 = model.predict((rgb_input, flow_input, edge_input))[0]
    print(pred2)
    label = 1 if pred >= 0.5 else 0
    return label, pred

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Classify a video as sludge (1) or non-sludge (0).")
    parser.add_argument("video_path", type=str, help="Path to the video file to classify")
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
        return

    try:
        label, confidence = classify_video(args.video_path)
        class_name = "Sludge" if label == 1 else "Non-sludge"
        print(f"Prediction: {class_name} (Label: {label}) with confidence {confidence:.4f}")
    except Exception as e:
        print(f"An error occurred during classification: {e}")

if __name__ == "__main__":
    main()
