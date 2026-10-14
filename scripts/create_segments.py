from pathlib import Path
import shutil
import subprocess
import pandas as pd


ROOT_M3AV = Path("data/processed/m3av_dataset")
MANIFEST_DIR = Path("data/manifests")
AUDIO_DIR = ROOT_M3AV / "dataset_v1.0_onlyaudio"
OUTPUT_DIR = ROOT_M3AV / "segments"


# AI code to crop audio to correct segments
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


def prepare_audio_split(split):
    manifest_path = MANIFEST_DIR / f"m3av_{split}_manifest.csv"
    output_dir = OUTPUT_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.is_file():
        raise ValueError(f"Manifest File not found. {manifest_path}")

    df = load_dataframe(manifest_path)
    
    success = 0
    skip = 0
    fail = []
    
    print(f"\nPreparing {split}: {len(df)} segments")
    print(f"Output directory: {output_dir}")
    
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        video_id = row["video_id"]
        sample_id = row["sample_id"]
        
        audio_path = AUDIO_DIR / f"{video_id}.flac"
        segment_path = output_dir / f"{sample_id}.wav"
        
        if i%100==0:
            print(f"[{i}/{len(df)}]. Current video id: {video_id}")
        
        # AI suggested this so we can reuse existing segments
        if segment_path.is_file() and segment_path.stat().st_size > 0:
            skip += 1
            continue
        
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            crop_audio(audio_path, row["start"], row["end"], segment_path)
            
            # AI suggestion to check that file exists and is not empty
            if not segment_path.is_file() or segment_path.stat().st_size == 0:
                raise RuntimeError("Output segment is missing or empty")

            success += 1
        
        except Exception as e:
            # Remove a potentially incomplete output file.
            segment_path.unlink(missing_ok=True) # AI suggestion to stop potential incomplete files
            
            print(f"ERROR: {sample_id}: {e}")
            fail.append({"sample_id": sample_id, "video_id": video_id, "error": str(e)})
        
    if fail:
        error_file = output_dir / "preparation_errors.csv"
        # .DataFrame ensure that the fail (list of dictionaries) is turned into a DataFrame before writing csv file
        pd.DataFrame(fail).to_csv(error_file, index=False)
        print(f"Errors saved to: {error_file}")
    
    print(f"\nFinished split: {split}")
    print(f"Number of newly created segment files: {success}")
    print(f"Number of already existing files: {skip}")
    print(f"Number of failed samples: {len(fail)}")


def main():
    splits = ["train", "dev", "test"]
    for split in splits:
        prepare_audio_split(split)


if __name__=="__main__":
    main()