import os
import whisper
import torch
import warnings # Import the warnings module

# Suppress the FP16 warning (OPTIONAL)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Define the paths (same as before)
sludge_audio_path = './dataset/audio/sludge'
non_sludge_audio_path = './dataset/audio/non_sludge'
text_sludge_path = './dataset/text/sludge'
text_non_sludge_path = './dataset/text/non_sludge'

# Create text directories if they don't exist
os.makedirs(text_sludge_path, exist_ok=True)
os.makedirs(text_non_sludge_path, exist_ok=True)

# Check for CUDA (GPU) availability
if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU (CUDA)")
else:
    device = "cpu"
    print("Using CPU (no GPU available)")

# Load the Whisper model (choose your model size)
model = whisper.load_model('base', device=device)

# Function to transcribe audio files (modified for error handling and clarity)
def transcribe_audio(audio_path, text_path):
    for filename in os.listdir(audio_path):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            audio_file = os.path.join(audio_path, filename)
            text_file = os.path.join(text_path, filename.replace('.wav', '.txt').replace('.mp3', '.txt'))

            # Skip if the text file already exists (avoid re-transcribing)
            if os.path.exists(text_file):
                print(f"Skipping {filename} (already transcribed)")
                continue

            try:
                # Transcribe audio
                result = model.transcribe(audio_file, language="Tagalog")  # Corrected language code
                transcription = result['text']

                # Save transcription to a text file
                with open(text_file, 'w', encoding='utf-8') as f:  # Specify UTF-8 encoding
                    f.write(transcription)
                print(f"Transcribed {filename} to {text_file}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


# --- Create dummy files for demonstration (IMPORTANT: Remove in real use) ---
os.makedirs(sludge_audio_path, exist_ok=True)
os.makedirs(non_sludge_audio_path, exist_ok=True)

# Create dummy .wav files (Whisper can handle empty WAV files for testing)
with open(os.path.join(sludge_audio_path, "sludge_example.wav"), "w") as f:
    pass  # Creates an empty file
with open(os.path.join(non_sludge_audio_path, "non_sludge_example.wav"), "w") as f:
    pass  # Creates an empty file
# --- End of dummy file creation ---

# Transcribe audio from both directories
transcribe_audio(sludge_audio_path, text_sludge_path)
transcribe_audio(non_sludge_audio_path, text_non_sludge_path)

print("Transcription complete.")
