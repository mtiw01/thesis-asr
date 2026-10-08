from pathlib import Path
import pandas as pd
import torch
import whisper

chosen_model = "whisper"

TEST_AUDIO_DIR = Path("data/processed/m3av_dataset/test_set_audio")

INPUT_CSV = Path("data/manifests/m3av_test_set_manifest.csv")

# load dataset
def load_dataframe():
    df = pd.read_csv(INPUT_CSV)
    print(f"Number of files/rows in manifest: {len(df)}")
    return df


def get_baseline(df):
    
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if chosen_model=="whisper":
        model = whisper.load_model("medium", device=device)
    elif chosen_model=="qwen":
        pass
    else:
        return "The chosen model is unknown."
    
    predictions = []
    for i, row in df.iterrows():
        
        sample_id = row["video_id"]
        audio_path = TEST_AUDIO_DIR / f"{sample_id}.flac"

        if not audio_path.exists():
            print(f"WARNING: audio not found: {audio_path}")
            predictions.append("")
            continue

        print(f"[{i + 1}/{len(df)}] {sample_id}")

        result = model.transcribe(str(audio_path), language="en")

        prediction = result["text"].strip()

        predictions.append(prediction)
    
    df['prediction'] = predictions
    
    file_path = f"outputs/baseline/{chosen_model}_predictions.csv"
    df.to_csv(file_path, index=False)
    print(f"The results are saved to: {file_path}")


def main():
    df = load_dataframe()
    get_baseline(df)
    print("Baseline predictions are complete")


if __name__=="__main__":
    main()