import json
from pathlib import Path

# Global Variables - Filepaths
ROOT_M3AV = Path("data/raw/m3av_dataset")
SET_SPLIT_JSON_FILE = ROOT_M3AV/"set_split.json"
DATASET_DIR = ROOT_M3AV/"dataset_v1.0_noaudio"
STATS_FILE = Path("outputs/m3av-noaudio-inspection-results.txt")

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
    stats = {
        split: {
            "videos": 0,
            "seconds": 0.0,
            "segments": 0,
            "words": 0,
        }
        for split in SET_SPLITS
    }
    
    for vid_id, split in splits.items():
        
        vid_id_dir = DATASET_DIR/vid_id
        transcript_file = vid_id_dir/"speech"/"final+timestamps.json"
        
        data = load_json(transcript_file)
        
        # get duration
        timestr = data[-1].get("timestr")
        _, end = timestr.split("_")
        end_seconds = int(end)/1000

        stats[split]["videos"] +=1
        stats[split]["seconds"] += end_seconds
        stats[split]["segments"] += len(data)
        stats[split]["words"] += sum(len(segment.get("words_spoken", [])) for segment in data)
    
    return stats


def dataset_stats(splits, stats):
    report = ""
    
    # datset info
    report+= "\n" + "=" * 100 + "\n" 
    report+= "M3AV DATASET\n" 
    report+= "=" * 100 + "\n"
    
    report+= f"\nSplit file: {SET_SPLIT_JSON_FILE}\n" 
    report+= f"Dataset directory: {DATASET_DIR}\n" 
    report+= f"Total entries: {len(splits)}\n"
    
    # summary 
    report+= "\n" + "=" * 100 + "\n" 
    report+= "DATASET SUMMARY\n" 
    report+= "=" * 100 + "\n" 
    report+= ( 
        f"{'Split':<10}" 
        f"{'Videos':>10}" 
        f"{'Hours':>12}" 
        f"{'Segments':>14}" 
        f"{'Words':>14}\n" 
    )
    
    report+= "-" * 60 + "\n" 
    
    for split in SET_SPLITS: 
        data = stats[split] 
        report += ( 
            f"{split:<10}" 
            f"{data['videos']:>10}" 
            f"{(data['seconds']/3600):>12.2f}" 
            f"{data['segments']:>14}" 
            f"{data['words']:>14}\n" 
        )
    
    # Detailed statistics 
    report+= "\n" + "=" * 100 + "\n" 
    report+= "SPLIT STATISTICS\n" 
    report+= "=" * 100 + "\n" 
    
    for split in SET_SPLITS: 
        data = stats[split]  
        
        report+= f"\n{split.upper()}\n"
        report+= "-" * 30 + "\n" 
        
        report+= f"Videos: {data['videos']}\n" 
        report+= f"Hours: {(data['seconds']/3600):.2f}\n"
        report+= f"Segments: {data['segments']}\n" 
        report+= f"Words: {data['words']}\n" 
        
        if data["segments"] > 0: 
            average_segment = data["seconds"] / data["segments"] 
            report+= f"Avg segment: {average_segment:.2f} seconds\n" 
        
        if data["videos"] > 0: 
            average_video = data["seconds"] / data["videos"] 
            report+= f"Avg video: {average_video / 60:.2f} minutes\n"
    
    report+= "\n" + "=" * 100
        
    # Print everything 
    print(report) 
    
    # Write report to file 
    with open(STATS_FILE, "w", encoding="utf-8") as f: 
        f.write(report)

    
def main():
    # check paths
    check_paths()
    
    # define splits
    set_splits = load_json(SET_SPLIT_JSON_FILE)
    
    # get stats
    stats = collect_statistics(set_splits)
    
    # print and write splits and stats
    dataset_stats(set_splits, stats)
    
    print("Done!")

if __name__=="__main__":
    main()