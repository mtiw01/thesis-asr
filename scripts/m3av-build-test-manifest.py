from pathlib import Path
import json
import csv


ROOT_M3AV = Path("data/raw/m3av_dataset")
SET_SPLIT_JSON_FILE = ROOT_M3AV / "set_split.json"
DATASET_DIR = ROOT_M3AV / "dataset_v1.0_noaudio"

OUTPUT_DIR = Path("data/manifests")
OUTPUT_FILE = OUTPUT_DIR / "m3av_test_set_manifest.csv"


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)


def load_test_video_ids(split_file):
    split_data = load_json(split_file)
    
    assert(isinstance(split_data, dict)), "m"


def main():
    test_video_ids = load_test_video_ids(SET_SPLIT_JSON_FILE)


if __name__=="__main__":
    main()