from pathlib import Path
import torch
import whisper
import pandas as pd
import json


# Configuration
CHOSEN_MODEL = "whisper" # whisper or qwen

if CHOSEN_MODEL=='whisper':
    SIZE = "turbo" # turbo or medium
    MODELS_DIR = Path(f"outputs/training/whisper_{SIZE}")
else:
    SIZE = "0.6B" # 1.7B or 0.6B
    MODELS_DIR = Path(f"outputs/training/qwen_{SIZE}")


# Constants
MODEL_NAME = "epoch2"

TEST_SEGMENT_DIR = Path("data/processed/m3av_dataset/segments/test")
TEST_MANIFEST = Path("data/manifests/m3av_test_manifest.csv")

OUTPUT_DIR = Path(f"outputs/training_word_boosting_evaluation")
VOCABULARY_DIR = Path("data/vocabulary_words10_ngram2.json")


# === Load data =================

def load_dataframe(input_csv):
    df = pd.read_csv(input_csv)
    required_columns = {"sample_id", "video_id", "start", "end", "reference"}
    
    missing = required_columns - set(df.columns)
    if missing: 
        raise ValueError(f"There are missing columns in manifest.\nMissing Columns: {missing}")
    if df["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample IDs.")
    
    print(f"Number of files/rows in manifest: {len(df)}")
    return df



def load_json(file):
    if not file.is_file():
        raise FileNotFoundError(
            f"Vocabulary file does not exist at the specified destination:\n    {file}"
        )
    
    with open(file, "r", encoding='utf-8') as f:
        res = json.load(f)

    if not isinstance(res, dict):
        raise ValueError("Vocabulary must be a dictionary.")
    
    vocabulary_data = {}
    for video_id, vocab in res.items():
        if not isinstance(vocab, list):
            raise ValueError(
                f"Vocabulary for all videos must be a list."
                f"Video ID: {video_id}."
                f"Vocabulary type: {type(vocab)}"
            )
        
        if not all(isinstance(word, str) for word in vocab):
            raise ValueError(
                f"Vocabulary values must be strings in a list."
                f"Current video_id: {video_id}"
            )
        
        
        original_vocab_len = len(vocab)
        vocab = list(dict.fromkeys(word.strip() for word in vocab if word.strip()))
        
        if not vocab:
            raise ValueError(f"Vocabulary for {video_id} is empty.")
        if len(vocab)!=original_vocab_len:
            print(f"Video ID: '{video_id}':")
            print(f"    {original_vocab_len} original values")
            print(f"    {len(vocab)} unique values")
        
        vocabulary_data[str(video_id)] = vocab
    
    print(f"\nLoaded vocabulary for {len(vocabulary_data)} videos.")
    
    return vocabulary_data


# === Model =====================

def load_model(size=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if CHOSEN_MODEL=="whisper":
        checkpoint_path = MODELS_DIR / f"{MODEL_NAME}.pt"

        print(f"Loading checkpoint: {checkpoint_path}")

        # Load the complete checkpoint.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True,)

        # Load the Whisper architecture.
        model = whisper.load_model(size, device=device)

        # Restore model weights.
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        return model, None
    
    elif CHOSEN_MODEL=="qwen":
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        if device!="cuda":
            raise RuntimeError("No CUDA GPU detected. For this full fine-tuning script, use a suitable GPU node.")
        
        checkpoint_path = MODELS_DIR / MODEL_NAME
        
        print(f"Using device: {device}")
        print(f"Loading checkpoint: {checkpoint_path}")
        
        processor = AutoProcessor.from_pretrained(checkpoint_path)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(checkpoint_path)
        
        model.to(device)

        model.eval()
        model.config.use_cache = True

        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = True
        
        return model, processor
    
    else: 
        raise ValueError(f"Unknown model: {CHOSEN_MODEL}")



def transcribe(model, processor, path, vocabulary):
    
    vocab_prompt = "Context for the audio: " + ", ".join(vocabulary)
    
    if CHOSEN_MODEL == "whisper":
        with torch.inference_mode():    
            result = model.transcribe(
                str(path),
                language="en",
                fp16=torch.cuda.is_available(),
                initial_prompt=vocab_prompt
            )
        return result["text"].strip()
    
    if CHOSEN_MODEL == "qwen":
        inputs = processor.apply_transcription_request(
            audio=str(path),
            language="English",
            prompt=vocab_prompt
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



def run_inference(df, vocabulary_video):
    
    manifest_videos = set(df["video_id"].astype(str))
    vocabulary_videos = set(vocabulary_video.keys())

    missing_videos = manifest_videos - vocabulary_videos

    if missing_videos:
        raise ValueError(
            f"Missing vocabulary for {len(missing_videos)} videos: "
            f"{sorted(missing_videos)}"
        )
    
    vocabulary_count = []
    vocabulary_used = []
    # load model
    model, processor = load_model(SIZE)
    
    predictions = []
    status = []
    
    for i, (_,row) in enumerate(df.iterrows(), start=1):
        sample_id = row["sample_id"]
        video_id = row["video_id"]
        segment_path = TEST_SEGMENT_DIR / f"{sample_id}.wav"
        
        vocabulary = vocabulary_video[str(video_id)]
        vocabulary_count.append(len(vocabulary))
        vocabulary_used.append(", ".join(vocabulary))
        
        print(f"[{i}/{len(df)}] {sample_id}")

        try:
            if not segment_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {segment_path}")
            
            # AI check to ensure that segment is not empty
            if segment_path.stat().st_size == 0:
                raise ValueError(
                    f"Test segment is empty: {segment_path}"
                )
            
            prediction = transcribe(model, processor, segment_path, vocabulary)

            predictions.append(prediction)
            status.append("ok")
            
        except Exception as e:
            print(f"ERROR: {sample_id}: {e}")
            predictions.append("")
            status.append(f"error: {e}")
    
    results = df.copy()
    results['model'] = CHOSEN_MODEL
    results['vocabulary_count'] = vocabulary_count
    results["vocabulary"] = vocabulary_used
    results['prediction'] = predictions
    results["status"] = status
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{CHOSEN_MODEL}_{SIZE}_{MODEL_NAME}_word_boosting_predictions.csv"
    results.to_csv(output_file, index=False)
    
    success_count = sum(s=="ok" for s in status)
    error_count = len(status) - success_count
    
    print("\n")
    print(f"Word Boosting complete.")
    print(f"Successful segments: {success_count}")
    print(f"Failed segments: {error_count}")
    print(f"Results saved to: {output_file}")


def main():
    df = load_dataframe(TEST_MANIFEST)#.head(10)
    vocabulary = load_json(VOCABULARY_DIR)
    run_inference(df, vocabulary)
    # start_time = time.perf_counter()

    # get_baseline(df)

    # end_time = time.perf_counter()
    # elapsed = end_time - start_time

    # print(f"\nTotal runtime: {elapsed:.2f} seconds")
    # print(f"Total runtime: {elapsed / 60:.2f} minutes")


if __name__=="__main__":
    main()
