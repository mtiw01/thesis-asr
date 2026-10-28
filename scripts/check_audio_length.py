
from pathlib import Path
import soundfile as sf

DATA_DIR = Path("data/processed/m3av_dataset/segments")
SPLITS = ["train", "dev"]
MAX_DURATION = 30.0

total = 0
errors = []

for split in SPLITS:
    split_dir = DATA_DIR / split
    wav_files = sorted(split_dir.glob("*.wav"))

    print(f"\nChecking {split}: {len(wav_files)} WAV files")

    for i, wav_path in enumerate(wav_files, start=1):
        total += 1

        try:
            info = sf.info(wav_path)
            duration = info.duration

            if duration > MAX_DURATION:
                errors.append((wav_path, duration, "TOO LONG"))

            # Flag files that are not exactly 30 seconds,
            # allowing a small tolerance for sample rounding.
            # elif abs(duration - MAX_DURATION) > 0.001:
            #     errors.append((wav_path, duration, "NOT EXACTLY 30s"))

        except Exception as exc:
            errors.append((wav_path, None, f"READ ERROR: {exc}"))
        
        if i%100==0:
            print(f"{i}/{len(wav_files)}. Error list length: {len(errors)}")

print(f"\nTotal WAV files checked: {total}")
print(f"Files flagged: {len(errors)}")

for path, duration, reason in errors[:50]:
    duration_text = (
        f"{duration:.6f}s" if duration is not None else "unknown duration"
    )
    print(f"{reason}: {path} — {duration_text}")

if len(errors) > 50:
    print(f"... and {len(errors) - 50} more")

if not errors:
    print("All WAV files are 30 seconds or less.")
