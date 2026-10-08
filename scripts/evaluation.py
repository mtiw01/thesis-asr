import pandas as pd
from jiwer import wer

chosen_model = "whisper" # whisper or qwen

file_path = f"outputs/baseline/{chosen_model}_predictions.csv"
df = pd.read_csv(file_path)

score = wer(df["reference"], df["prediction"])

print(f"Model: {chosen_model}; WER: {score:.4f}")