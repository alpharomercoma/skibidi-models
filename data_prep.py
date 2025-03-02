import os
import glob
# from moviepy.editor import VideoFileClip, ColorClip, concatenate_videoclips
from moviepy import *

import librosa
import whisper

# Define target duration based on industry standards for short-form content (~30 seconds)
TARGET_DURATION = 30.0  # seconds

# Define input and output directories
base_input = './dataset/video'
base_output_audio = './new_dataset/audio'
base_output_video = './new_dataset/video_norm'
base_output_text = './new_dataset/text'
categories = ['sludge', 'non_sludge']

# Create output directories if they do not exist
for base in [base_output_audio, base_output_video, base_output_text]:
    for cat in categories:
        os.makedirs(os.path.join(base, cat), exist_ok=True)

# Load Whisper model for transcription
model = whisper.load_model("large")

def process_video_clip(clip, target_duration=TARGET_DURATION):
    """
    Process a video clip by trimming it to the target duration if it's longer,
    or padding it with a black clip if it's shorter.
    """
    if clip.duration < target_duration:
        # Calculate remaining duration to pad
        remaining = target_duration - clip.duration
        # Create a black clip of the required duration with the same size and FPS
        black_clip = ColorClip(size=clip.size, color=(0, 0, 0), duration=remaining)
        black_clip = black_clip.with_fps(clip.fps)
        # Concatenate the original clip with the black clip
        new_clip = concatenate_videoclips([clip, black_clip])
    else:
        # If longer, take only the first target_duration seconds
        new_clip = clip.subclipped(0, target_duration)
    return new_clip

# Initialize counters for normalized file naming
counter = {"sludge": 378, "non_sludge": 1}

for cat in categories:
    input_dir = os.path.join(base_input, cat)
    video_files = sorted(glob.glob(os.path.join(input_dir, "*.*")))

    for video_file in video_files:
        print(f"Processing {video_file}")

        # Load the video file
        clip = VideoFileClip(video_file)

        # Normalize video duration (pad or trim to TARGET_DURATION)
        clip = process_video_clip(clip, TARGET_DURATION)

        # Generate a normalized file name (e.g., "sludge_001.mp4")
        new_filename = f"{cat}_{counter[cat]:03d}.mp4"
        counter[cat] += 1

        # Save the normalized video in the output directory
        video_save_path = os.path.join(base_output_video, cat, new_filename)
        clip.write_videofile(video_save_path, codec='libx264', audio_codec='aac')

        # Extract audio and save as WAV for Whisper transcription
        audio_save_filename = new_filename.replace('.mp4', '.wav')
        audio_save_path = os.path.join(base_output_audio, cat, audio_save_filename)
        clip.audio.write_audiofile(audio_save_path)

        # Load the audio using librosa (if further processing is needed)
        # y, sr = librosa.load(audio_save_path, sr=None)

        # Transcribe the audio using Whisper
        result = model.transcribe(audio_save_path)
        transcript = result["text"]

        # Save the transcript into the designated text output directory
        text_save_filename = new_filename.replace('.mp4', '.txt')
        text_save_path = os.path.join(base_output_text, cat, text_save_filename)
        with open(text_save_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        # Close the clip to free resources
        clip.close()
