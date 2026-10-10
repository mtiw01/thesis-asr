from pathlib import Path
import pandas as pd
import torch
import whisper
import tempfile
import subprocess
import time


# Configuration
CHOSEN_MODEL = "qwen" # whisper or qwen

WHISPER_SIZE = "medium"
QWEN_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"


# Constants
TEST_AUDIO_DIR = Path("data/processed/m3av_dataset/test_set_audio")
INPUT_CSV = Path("data/manifests/m3av_test_manifest.csv")
OUTPUT_DIR = Path("outputs/baseline")


# Load dataset
def load_dataframe():
    df = pd.read_csv(INPUT_CSV)
    required_columns = {
            "sample_id",
            "video_id",
            "start",
            "end",
            "reference",
        }
    
    missing = required_columns - set(df.columns)
    if missing: 
        raise ValueError(f"There are missing columns in manifest.\nMissing Columns: {missing}")
    if df["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample IDs.")
    
    print(f"Number of files/rows in manifest: {len(df)}")
    return df


def load_model():
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
        
    if CHOSEN_MODEL=="whisper":
        model = whisper.load_model(WHISPER_SIZE, device=device)
        return model, None
    elif CHOSEN_MODEL=="qwen":
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        model = AutoModelForMultimodalLM.from_pretrained(
            QWEN_MODEL_ID,
                device_map="auto",
        )
        model.eval()
        return model, processor
    else:
        raise ValueError(f"Unknown model: {CHOSEN_MODEL}")


def transcribe(model, processor, path):
    if CHOSEN_MODEL == "whisper":
        result = model.transcribe(
            str(path),
            language="en",
            fp16=torch.cuda.is_available(),
        )
        return result["text"].strip()
    
    if CHOSEN_MODEL == "qwen":
        inputs = processor.apply_transcription_request(
            audio=str(path),
            language="English"
        ).to(model.device, model.dtype)
        
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        generated_ids = output_ids[
            :, inputs["input_ids"].shape[1]:
        ]

        return processor.decode(
            generated_ids,
            return_format="transcription_only",
        )[0].strip()

    raise ValueError(f"Unknown model: {CHOSEN_MODEL}")
        


def crop_audio(audio_path, start, end, segment_path):
    """Extract one segment from the full-video audio using FFmpeg."""
    duration = float(end) - float(start)

    if duration <= 0:
        raise ValueError(f"Invalid segment duration: {start} to {end}")

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", str(float(start)),
        "-i", str(audio_path),
        "-t", str(duration),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(segment_path),
    ]

    subprocess.run(command, check=True)


def get_baseline(df):
    
    # load model
    model, processor = load_model()
    
    predictions = []
    statuses = []
    
    with tempfile.TemporaryDirectory(prefix="m3av_segments_") as temp_dir:
        temp_dir = Path(temp_dir)
        
        for i, (_,row) in enumerate(df.iterrows(), start=1):
            video_id = row["video_id"]
            sample_id = row["sample_id"]
            audio_path = TEST_AUDIO_DIR / f"{video_id}.flac"
            segment_path = temp_dir / f"{sample_id}.wav"
            
            print(f"[{i}/{len(df)}] {sample_id}")

            try:
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                    # print(f"WARNING: audio not found: {audio_path}")
                    # predictions.append("")
                    # continue
                
                crop_audio(audio_path, row["start"], row["end"], segment_path)
                
                prediction = transcribe(model, processor, segment_path)

                predictions.append(prediction)
                statuses.append("ok")
            
            except Exception as e:
                print(f"ERROR: {sample_id}: {e}")
                predictions.append("")
                statuses.append(f"error: {e}")
    
    results = df.copy()
    results['model'] = CHOSEN_MODEL
    results['prediction'] = predictions
    results["status"] = statuses
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{CHOSEN_MODEL}_predictions.csv"
    results.to_csv(output_file, index=False)
    print(f"The results are saved to: {output_file}")
    
    success_count = sum(status == "ok" for status in statuses)
    error_count = len(statuses) - success_count
    
    print("\nBaseline inference complete.")
    print(f"Successful segments: {success_count}")
    print(f"Failed segments: {error_count}")
    print(f"Results saved to: {output_file}")


def main():
    df = load_dataframe() #.head(10)
    # get_baseline(df)
    start_time = time.perf_counter()

    get_baseline(df)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"\nTotal runtime: {elapsed:.2f} seconds")
    print(f"Total runtime: {elapsed / 60:.2f} minutes")


if __name__=="__main__":
    main()