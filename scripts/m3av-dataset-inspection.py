import json
from pathlib import Path

# Global Variables - Filepaths
ROOT_M3AV = Path("data/raw/m3av_dataset")
SET_SPLIT_JSON_FILE = ROOT_M3AV/"set_split.json"
DATASET_DIR = ROOT_M3AV/"dataset_v1.0_noaudio"

SET_SPLITS = ["train", "test", "dev"]

def check_paths():
    if not SET_SPLIT_JSON_FILE.exists():
        raise FileNotFoundError(f"The file set_split.json was not found in the expected folder {ROOT_M3AV}.")

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset Directory not found: {DATASET_DIR}")


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_statistics(splits):
    pass


def dataset_stats(splits, stats):
    res = 0

    
def main():
    # check paths
    check_paths()
    
    # define splits
    set_splits = load_json(SET_SPLIT_JSON_FILE)
    print(set_splits)
    
    # get stats
    stats = collect_statistics(set_splits)
    
    # print and write splits and stats
    dataset_stats(set_splits, stats)
    
    print("Done!")

if __name__=="__main__":
    main()