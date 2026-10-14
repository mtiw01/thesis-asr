from pathlib import Path
import json
import shutil


ROOT_M3AV = Path("data/processed/m3av_dataset")
SET_SPLIT_JSON_FILE = ROOT_M3AV / "set_split.json"
DATASET_DIR_AUDIO = ROOT_M3AV / "dataset_v1.0_onlyaudio"

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)


def create_folder(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)


def move_files(output_dir, split_ids):
    copied = 0
    missing = []

    for i, video_id in enumerate(split_ids, start=1):
        file_name = f"{video_id}.flac"
        source = DATASET_DIR_AUDIO / file_name
        target = output_dir / file_name
        if i%10==0:
            print(f"{i}/{len(split_ids)}")
        if source.exists():
            shutil.copy(source, target)
            copied += 1
        else:
            missing.append(video_id)

    return copied, missing


def main():
    splits = ["train", "dev", "test"]
    for split in splits:
        output_dir = ROOT_M3AV / f"{split}_set_audio"
        print(output_dir)
        set_splits = load_json(SET_SPLIT_JSON_FILE)
        split_ids = [
                video_id
                for video_id, s in set_splits.items()
                if s == split
            ]
        print(split_ids[:10])
        create_folder(output_dir)
        print("Folder created.")
        copied, missing = move_files(output_dir, split_ids)
        
        print(f"Folder created and files copied successfully.")
        print(f"Copied files:  {copied}")
        print(f"Missing files: {missing}")


if __name__=="__main__":
    main()