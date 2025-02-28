import os
from moviepy import VideoFileClip
import librosa
import soundfile as sf

def process_video_directory(video_dir, audio_dir):
    """
    Processes all video files in a given directory:
      - Extracts the raw audio and saves it as <video_basename>_raw.wav.
      - Processes the raw audio to generate a transcription-ready audio file
        (resampled to 16 kHz, mono) and saves it as <video_basename>_transcribe.wav.
    """
    os.makedirs(audio_dir, exist_ok=True)
<<<<<<< HEAD
    
=======

>>>>>>> 97690b0 (feat: add v4 & v5)
    for filename in os.listdir(video_dir):
        # Filter for typical video file extensions; adjust if needed
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_dir, filename)
            base_name = os.path.splitext(filename)[0]
<<<<<<< HEAD
            
            # Define output paths for the two audio files
            raw_audio_path = os.path.join(audio_dir, base_name + "_raw.wav")
            transcribe_audio_path = os.path.join(audio_dir, base_name + "_transcribe.wav")
            
            print(f"Processing video: {video_path}")
            
=======

            # Define output paths for the two audio files
            raw_audio_path = os.path.join(audio_dir, base_name + "_raw.wav")
            transcribe_audio_path = os.path.join(audio_dir, base_name + "_transcribe.wav")

            print(f"Processing video: {video_path}")

>>>>>>> 97690b0 (feat: add v4 & v5)
            # --- 1. Extract raw audio from video ---
            try:
                clip = VideoFileClip(video_path)
                # Write the audio to a WAV file using a standard PCM codec
                clip.audio.write_audiofile(raw_audio_path, codec='pcm_s16le')
                clip.close()
            except Exception as e:
                print(f"Error extracting audio from {video_path}: {e}")
                continue
<<<<<<< HEAD
            
=======

>>>>>>> 97690b0 (feat: add v4 & v5)
            # --- 2. Process audio for transcription (Whisper) ---
            try:
                # Load raw audio with librosa. Using sr=None preserves the original sampling rate.
                y, sr = librosa.load(raw_audio_path, sr=None, mono=True)
<<<<<<< HEAD
                
=======

>>>>>>> 97690b0 (feat: add v4 & v5)
                # Whisper is trained on 16 kHz audio. Resample if needed.
                if sr != 16000:
                    y_transcribe = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    sr_transcribe = 16000
                else:
                    y_transcribe = y
                    sr_transcribe = sr
<<<<<<< HEAD
                
=======

>>>>>>> 97690b0 (feat: add v4 & v5)
                # Write out the transcription-ready audio file.
                sf.write(transcribe_audio_path, y_transcribe, sr_transcribe)
            except Exception as e:
                print(f"Error processing audio from {raw_audio_path}: {e}")

if __name__ == "__main__":
    # Define video input directories and corresponding audio output directories
<<<<<<< HEAD
    video_sludge_dir = './dataset/video/sludge'
    video_non_sludge_dir = './dataset/video/non_sludge'
    audio_sludge_dir = './dataset/audio/sludge'
    audio_non_sludge_dir = './dataset/audio/non_sludge'
    
=======
    video_sludge_dir = '../dataset/video/sludge'
    video_non_sludge_dir = '../dataset/video/non_sludge'
    audio_sludge_dir = '../dataset/audio/sludge'
    audio_non_sludge_dir = '../dataset/audio/non_sludge'

>>>>>>> 97690b0 (feat: add v4 & v5)
    # Process both categories
    process_video_directory(video_sludge_dir, audio_sludge_dir)
    process_video_directory(video_non_sludge_dir, audio_non_sludge_dir)
