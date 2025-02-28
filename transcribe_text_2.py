import os
import jax
import jax.numpy as jnp
from whisper_jax import FlaxWhisperPipline

def transcribe_audio_files(input_dir, output_dir, pipeline):
    """
    Transcribes only audio files in `input_dir` whose names end with '_transcribe.wav'.
    The resulting text is saved (with the same base name but with a .txt extension) in `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
<<<<<<< HEAD
    
=======

>>>>>>> 97690b0 (feat: add v4 & v5)
    # Process only files that end with '_transcribe.wav'
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('_transcribe.wav'):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(output_dir, output_filename)
<<<<<<< HEAD
            
=======

>>>>>>> 97690b0 (feat: add v4 & v5)
            print(f"Transcribing {input_path} on TPU...")
            try:
                # The pipeline uses batching and JAX's pmap to leverage TPU cores.
                # We set task="transcribe" to output the transcription in the original language.
                result = pipeline(input_path, task="transcribe", return_timestamps=False)
                text = result["text"]
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Saved transcription to {output_path}")
            except Exception as e:
                print(f"Error transcribing {input_path}: {e}")

def main():
    # Define categories and directories
    categories = ["sludge", "non_sludge"]
<<<<<<< HEAD
    audio_base_dir = "./dataset/audio"
    text_base_dir = "./dataset/text"
=======
    audio_base_dir = "../dataset/audio"
    text_base_dir = "../dataset/text"
>>>>>>> 97690b0 (feat: add v4 & v5)

    # Verify TPU devices are visible via JAX
    print("JAX devices:", jax.devices())

    # Load the Whisper JAX pipeline.
    # Here we choose the "openai/whisper-large-v2" model, which in JAX is optimized for TPU.
    # Using jnp.bfloat16 (preferred on TPU v4 or TPU VMs) and enabling batching for speed.
    pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)
<<<<<<< HEAD
    
=======

>>>>>>> 97690b0 (feat: add v4 & v5)
    # Process each category directory
    for cat in categories:
        input_dir = os.path.join(audio_base_dir, cat)
        output_dir = os.path.join(text_base_dir, cat)
        transcribe_audio_files(input_dir, output_dir, pipeline)

if __name__ == "__main__":
    main()
