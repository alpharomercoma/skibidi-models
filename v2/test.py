import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

# Updated custom edge extraction function to return a single-channel edge map.
@register_keras_serializable()
def edge_extraction(x):
    # x: shape (H, W, C)
    sobel = tf.image.sobel_edges(x)  # returns shape (H, W, C, 2)
    # Compute the gradient magnitude for each channel.
    mag = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))  # shape: (H, W, C)
    # Average over the channels to obtain a single channel.
    gray = tf.reduce_mean(mag, axis=-1, keepdims=True)  # shape: (H, W, 1)
    return gray

# Define and register the custom SpatialAttention layer.
@register_keras_serializable()
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Create a convolutional layer with one filter for the attention map.
        self.conv = tf.keras.layers.Conv2D(
            filters=1, kernel_size=self.kernel_size, padding="same", activation="sigmoid"
        )
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        # Compute average and max pooling along the channel dimension.
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config

# PARAMETERS
NUM_FRAMES = 16         # fixed number of frames per video
IMG_SIZE = 224          # height and width of each frame
VIDEO_PATH = "/home/alpha/alpha/manual/non_sludge_5.mp4"
MODEL_PATH = "final_video_classifier_model.h5"

def _load_video(path):
    """
    Load a video file and uniformly sample NUM_FRAMES frames.
    Frames are resized, converted to RGB, and normalized.
    Limits video duration to 40 seconds.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")

    # Get frames per second; if unavailable, assume 30 fps.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    # Limit video duration to 40 seconds.
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
            # Resize, convert BGR -> RGB, and normalize.
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype('float32') / 255.0
            extracted[frame_id] = frame
        frame_id += 1
    cap.release()

    # If no frames were extracted, return an array of zeros.
    if not extracted:
        return np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    # If a frame is missing, pad using the last available frame.
    last_available = max(extracted.keys())
    out_frames = []
    for idx in indices:
        if idx in extracted:
            out_frames.append(extracted[idx])
        else:
            out_frames.append(extracted[last_available])
    return np.array(out_frames)

def main():
    # Check if the model file exists.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}. Please verify that the path is correct and the model has been saved."
        )

    # Load the trained model with the custom objects.
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'edge_extraction': edge_extraction,
            'SpatialAttention': SpatialAttention
        }
    )

    # Preprocess the video.
    video = _load_video(VIDEO_PATH)
    # Expand dimensions to add a batch dimension: (1, NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    video_batch = np.expand_dims(video, axis=0)

    # Get prediction from the model.
    prediction = model.predict(video_batch)
    # For binary classification, the output is a sigmoid probability.
    confidence = prediction[0][0]
    # Assuming label 0: non_sludge and label 1: sludge.
    label = "sludge" if confidence >= 0.5 else "non_sludge"

    print(f"Video: {VIDEO_PATH}")
    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    main()
