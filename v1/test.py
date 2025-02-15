#!/usr/bin/env python3
import os
# Force CPU usage to avoid CUDA initialization errors.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# -------------------------------
# Updated Custom Layer: Spatial Attention
# -------------------------------
class SpatialAttention(keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = keras.layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal'
        )

    def build(self, input_shape):
        # The input will have shape (batch, H, W, C). After concatenating along channels, shape becomes (batch, H, W, 2).
        self.conv.build((None, input_shape[1], input_shape[2], 2))
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape.
        return input_shape

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config

# -------------------------------
# Video Loading Function
# -------------------------------
def load_video(path, num_frames=16, img_size=224):
    cap = cv2.VideoCapture(path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = num_frames
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
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])
    return np.array(frames)

# -------------------------------
# Main Prediction Function
# -------------------------------
def main():
    video_path = "/home/alpha/alpha/manual/sludge.mp4"
    print(f"Loading video: {video_path}")
    video = load_video(video_path)
    video_batch = np.expand_dims(video, axis=0)

    # Load the saved model with the updated custom layer.
    try:
        model = keras.models.load_model(
            'video_classification_sludge_model.h5',
            custom_objects={'SpatialAttention': SpatialAttention},
            compile=False
        )
    except ValueError as e:
        print("Error loading model:", e)
        print("Tip: Consider re-saving your model using the SavedModel format to avoid custom layer serialization issues.")
        return

    prediction = model.predict(video_batch)
    confidence = float(prediction[0])
    label = "sludge" if confidence >= 0.5 else "non_sludge"
    print(f"Prediction: {label} with confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    main()
