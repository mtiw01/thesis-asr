import pandas as pd
from jiwer import process_words, visualize_alignment, wer
import re
from pathlib import Path

CHOSEN_MODEL = "qwen" # whisper or qwen

if CHOSEN_MODEL=='whisper':
    SIZE = "turbo" # turbo or medium
else:
    SIZE = "0.6B" # 1.7B or 0.6B
    QWEN_MODEL_ID = f"Qwen/Qwen3-ASR-{SIZE}-hf" # Qwen/Qwen3-ASR-1.7B-hf or Qwen/Qwen3-ASR-0.6B-hf

OUTPUT_DIR = Path("outputs/baseline")

# AI suggested normalization. Mine were only the apostrophe replacement and the .strip call
def normalize_text(text):
    text = text.lower()
    # remove apostrophes without splitting contractions
    text = text.replace("'", "")
    # treat hyphens and dashes as word separators
    text = re.sub(r"[-‐‑‒–—]", " ", text)
    # replace remaining punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def get_score(ref, pred):

    # Use your normalized transcripts
    output = process_words(ref, pred)
    
    res = ''
    
    score = wer(ref, pred)
    # res += f"-------------------------"
    res += f"=== WER Model: {CHOSEN_MODEL.upper()} ==="
    res += f"\n-----------------------------\n"
    res += f"WER: {score:.4f}\n\n"
    res += "="*75 + "\n\n\n"

    # Display the alignment and errors
    res += (visualize_alignment(
        output,
        show_measures=True,
        skip_correct=True,
        line_width=100
    ))
    
    return res


def write_result(res):
    output_file = OUTPUT_DIR / f"{CHOSEN_MODEL}_{SIZE}_wer_evaluation.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(res)
    return output_file


def main():
    
    if CHOSEN_MODEL not in ['whisper', 'qwen']:
        return "Unknown Model"
    
    # set filepath
    file_path = OUTPUT_DIR / f"{CHOSEN_MODEL.lower()}_{SIZE}_predictions.csv"
    # read csv file
    df = pd.read_csv(file_path)
    
    # extract reference and prediction as list of strings
    reference = df["reference"].fillna("").astype(str).tolist()
    prediction = df["prediction"].fillna("").astype(str).tolist()
    
    # normalize both
    ref = []
    pred = []
    for i in range(len(reference)):
        ref.append(normalize_text(reference[i]))
        pred.append(normalize_text(prediction[i]))

    res = get_score(ref, pred)
    
    output_file = write_result(res)
    
    print(f"\nDone. File can be found in '{output_file}'.\n")


if __name__=="__main__":
    main()
    
#score = wer(ref, pred)

#print(f"Model: {CHOSEN_MODEL}; WER: {score:.4f}")