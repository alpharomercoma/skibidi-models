import os
import cv2
import numpy as np
import tensorflow as tf

# PARAMETERS (must match training settings)
NUM_FRAMES = 16
IMG_SIZE = 224

# Register the custom function so Keras can deserialize the Lambda layer
@tf.keras.utils.register_keras_serializable()
def compute_edge_magnitude(x):
    # x has shape (batch, H, W, 3)
    sobel = tf.image.sobel_edges(x)  # shape: (batch, H, W, 3, 2)
    # Compute gradient magnitude for each channel
    mag = tf.sqrt(tf.reduce_sum(tf.square(sobel), axis=-1))  # (batch, H, W, 3)
    # Combine across RGB channels (using mean here)
    mag = tf.reduce_mean(mag, axis=-1, keepdims=True)  # (batch, H, W, 1)
    return mag

def _load_video(path):
    """
    Loads a video file and extracts a fixed number of frames (NUM_FRAMES)
    resized to (IMG_SIZE, IMG_SIZE) and normalized to [0,1].
    """
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

def main():
    # Path to the video file
    video_path = "/home/alpha/alpha/Trial Dataset 500/video_1.mp4"

    # Load the saved model, providing the custom object for the Lambda layer
    model = tf.keras.models.load_model(
        'final_video_classifier_model_2.h5',
        custom_objects={'compute_edge_magnitude': compute_edge_magnitude}
    )

    # Load and preprocess video frames
    video_frames = _load_video(video_path)
    # Add batch dimension: expected input shape is (batch, NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    video_frames = np.expand_dims(video_frames, axis=0)

    # Run inference on the video
    prediction = model.predict(video_frames)

    # For binary classification using sigmoid activation:
    # A value >= 0.5 indicates sludge (label 1), otherwise non-sludge (label 0)
    confidence = prediction[0][0]
    label = "Sludge" if confidence >= 0.5 else "Non-sludge"

    print(f"Predicted label: {label}")
    print(f"Confidence level: {confidence:.2f}")

if __name__ == "__main__":
    main()
