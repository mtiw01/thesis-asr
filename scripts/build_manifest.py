from pathlib import Path
import json
import csv


ROOT_M3AV = Path("data/processed/m3av_dataset")
SET_SPLIT_JSON_FILE = ROOT_M3AV / "set_split.json"
DATASET_DIR = ROOT_M3AV / "dataset_v1.0_noaudio"

OUTPUT_DIR = Path("data/manifests")



def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_timestr(timestr):
    """
    Convert timestr from milliseconds to seconds.
    Example:
        "0004960_0013700" -> (4.96, 13.70)
    """
    start, end = timestr.split("_")

    start_seconds = int(start) / 1000.0
    end_seconds = int(end) / 1000.0

    return start_seconds, end_seconds


def load_video_ids(split_file):
    split_data = load_json(split_file)
    
    if not isinstance(split_data, dict):
        raise ValueError(
            f"Expected split file to contain a JSON object."
        )
    
    return split_data


def build_manifest(split, video_ids):
    rows = []
    
    for video_id in video_ids:
        video_dir = DATASET_DIR / video_id
        speech_file = video_dir / "speech" / "final+timestamps.json"
        
        if not video_dir.exists():
            raise FileNotFoundError(
                f"Video directory does not exist: {video_dir}"
            )

        if not speech_file.exists():
            raise FileNotFoundError(
                f"Speech file does not exist: {speech_file}"
            )

        segments = load_json(speech_file)

        if not isinstance(segments, list):
            raise ValueError(
                f"Expected list of segments in {speech_file}, "
                f"got {type(segments).__name__}"
            )
        
        for segment_index, segment in enumerate(segments):
            required_fields = {
                "timestr",
                "final_spoken",
                "final_written",
            }

            missing = required_fields - segment.keys()

            if missing:
                raise ValueError(
                    f"{speech_file}, segment {segment_index} "
                    f"is missing fields: {sorted(missing)}"
                )

            start, end = parse_timestr(segment["timestr"])

            if end <= start:
                raise ValueError(
                    f"Invalid segment timing in {speech_file}, "
                    f"segment {segment_index}: "
                    f"{segment['timestr']}"
                )

            sample_id = f"{video_id}_{segment_index:06d}"
            
            rows.append(
                {
                    "sample_id": sample_id,
                    "video_id": video_id,
                    "split": split,
                    "segment_index": segment_index,
                    "start": start,
                    "end": end,
                    "reference": segment["final_spoken"],
                    "reference_written": segment["final_written"],
                }
            )
    
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "video_id",
        "split",
        "segment_index",
        "start",
        "end",
        "reference",
        "reference_written",
    ]
    
    output_file = OUTPUT_DIR / f"m3av_{split}_manifest.csv"
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{split.upper()} Manifest created successfully.")
    print(f"Output file: {output_file}")
    print(f"Videos:      {len(video_ids)}")
    print(f"Segments:    {len(rows)}")


def main():
    splits = ['train', 'test', 'dev']
    video_ids = load_video_ids(SET_SPLIT_JSON_FILE)
    
    for split in splits:
        ids = sorted([
                video_id
                for video_id, s in video_ids.items()
                if s == split
            ])
        build_manifest(split, ids)


if __name__=="__main__":
    main()