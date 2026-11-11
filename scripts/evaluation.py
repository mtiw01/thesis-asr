import pandas as pd
from jiwer import process_words, visualize_alignment, wer
from pathlib import Path
from normalize_text import normalize_text

INFERENCE_TYPE = "training_word_boosting" # "baseline" , "baseline_word_boosting" , "training" , "training_word_boosting"
CHOSEN_MODEL = "whisper" # whisper or qwen

if CHOSEN_MODEL=='whisper':
    SIZE = "medium" # turbo or medium
else:
    SIZE = "1.7B" # 1.7B or 0.6B

OUTPUT_DIR = Path(f"outputs/{INFERENCE_TYPE}_evaluation")


def get_score(ref, pred):

    # Use your normalized transcripts
    output = process_words(ref, pred)
    
    score = output.wer
    
    res = ''
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
    if INFERENCE_TYPE=="baseline":
        output_file = OUTPUT_DIR / f"{CHOSEN_MODEL}_{SIZE}_wer_evaluation.txt"
    elif INFERENCE_TYPE=="baseline_word_boosting":
        output_file = OUTPUT_DIR / f"{CHOSEN_MODEL}_{SIZE}_word_boosting_wer_evaluation.txt"
    elif INFERENCE_TYPE=="training":
        output_file = OUTPUT_DIR / f"{CHOSEN_MODEL}_{SIZE}_epoch2_wer_evaluation.txt"
    elif INFERENCE_TYPE=="training_word_boosting":
        output_file = OUTPUT_DIR / f"{CHOSEN_MODEL}_{SIZE}_epoch2_word_boosting_wer_evaluation.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(res)
    return output_file


def main():
    
    if CHOSEN_MODEL not in ['whisper', 'qwen']:
        return "Unknown Model"
    
    # set filepath
    if INFERENCE_TYPE=="baseline":
        file_path = OUTPUT_DIR / f"{CHOSEN_MODEL.lower()}_{SIZE}_predictions.csv"
    if INFERENCE_TYPE=="baseline_word_boosting":
        file_path = OUTPUT_DIR / f"{CHOSEN_MODEL.lower()}_{SIZE}_predictions.csv"
    if INFERENCE_TYPE=="training":
        file_path = OUTPUT_DIR / f"{CHOSEN_MODEL.lower()}_{SIZE}_epoch2_predictions.csv"
    if INFERENCE_TYPE=="training_word_boosting":
        file_path = OUTPUT_DIR / f"{CHOSEN_MODEL.lower()}_{SIZE}_epoch2_word_boosting_predictions.csv"
    
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