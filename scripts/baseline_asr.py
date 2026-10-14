from pathlib import Path
import pandas as pd
import torch
import whisper
import time
from create_segments import load_dataframe


# Configuration
CHOSEN_MODEL = "qwen" # whisper or qwen

if CHOSEN_MODEL=='whisper':
    SIZE = "turbo" # turbo or medium
else:
    SIZE = "0.6B" # 1.7B or 0.6B
    QWEN_MODEL_ID = f"Qwen/Qwen3-ASR-{SIZE}-hf" # Qwen/Qwen3-ASR-1.7B-hf or Qwen/Qwen3-ASR-0.6B-hf


# Constants
TEST_SEGMENT_DIR = Path("data/processed/m3av_dataset/segments/test")
INPUT_CSV = Path("data/manifests/m3av_test_manifest.csv")
OUTPUT_DIR = Path("outputs/baseline")


def load_model():
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
        
    if CHOSEN_MODEL=="whisper":
        model = whisper.load_model(SIZE, device=device)
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


def get_baseline(df):
    
    # load model
    model, processor = load_model()
    
    predictions = []
    status = []
    
    for i, (_,row) in enumerate(df.iterrows(), start=1):
        sample_id = row["sample_id"]
        segment_path = TEST_SEGMENT_DIR / f"{sample_id}.wav"
        
        print(f"[{i}/{len(df)}] {sample_id}")

        try:
            if not segment_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {segment_path}")
            
            # AI check to ensure that segment is not empty
            if segment_path.stat().st_size == 0:
                raise ValueError(
                    f"Test segment is empty: {segment_path}"
                )
            
            prediction = transcribe(model, processor, segment_path)

            predictions.append(prediction)
            status.append("ok")
            
        except Exception as e:
            print(f"ERROR: {sample_id}: {e}")
            predictions.append("")
            status.append(f"error: {e}")
    
    results = df.copy()
    results['model'] = CHOSEN_MODEL
    results['prediction'] = predictions
    results["status"] = status
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{CHOSEN_MODEL}_{SIZE}_predictions.csv"
    results.to_csv(output_file, index=False)
    
    success_count = sum(s=="ok" for s in status)
    error_count = len(status) - success_count
    
    print("\nBaseline complete.")
    print(f"The results are saved to: {output_file}")
    print(f"Successful segments: {success_count}")
    print(f"Failed segments: {error_count}")
    print(f"Results saved to: {output_file}")


def main():
    df = load_dataframe(INPUT_CSV) #.head(10)
    get_baseline(df)
    # start_time = time.perf_counter()

    # get_baseline(df)

    # end_time = time.perf_counter()
    # elapsed = end_time - start_time

    # print(f"\nTotal runtime: {elapsed:.2f} seconds")
    # print(f"Total runtime: {elapsed / 60:.2f} minutes")


if __name__=="__main__":
    main()