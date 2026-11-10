from pathlib import Path
import json
import pandas as pd
import yake
import re


# Constants
M3AV_DATASET_DIR = Path("data/processed/m3av_dataset/dataset_v1.0_noaudio")
TEST_MANIFEST = Path("data/manifests/m3av_test_manifest.csv")

OUTPUT_DIR = Path(f"data/vocabulary.json")

MAX_CANDIDATES = 50
MAX_NGRAM_SIZE = 3


# === Load data =================

def load_csv(file):
    if not file.is_file():
        raise FileNotFoundError(f"File was not found at the specified location.\nLocation: {file}")
    
    df = pd.read_csv(file)
    required_columns = {"sample_id", "video_id", "start", "end", "reference"}
    
    missing = required_columns - set(df.columns)
    if missing: 
        raise ValueError(f"There are missing columns in manifest.\nMissing Columns: {missing}")
    if df["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample IDs.")
    
    print(f"Number of samples in manifest: {len(df)}")
    return df


# === load ocr data =============
def load_ocr_transcriptions(video_id):
    
    ocr_file = Path(f'{video_id}/ocr/ocr_result.json')
    
    if not ocr_file.is_file():
        raise FileNotFoundError(f"OCR file was not found at the specified location.\nLocation: {ocr_file}")
    
    with open(ocr_file, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    
    if not isinstance(ocr_data, list):
        raise ValueError(f"Expected a list in {ocr_file}.")
    
    texts = []
    
    for data in ocr_data:
        for image in data["ocr_data"]:
            if not isinstance(image, dict):
                continue
        
            text = image.get("transcription", "") # use get to avoid errors for missing values
        
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        
    if not texts:
        raise ValueError(
            f"No OCR transcription found in {ocr_file}"
        )

    return "\n".join(texts)



# === save json file ==========

def save_vocab(vocabulary, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, indent=4, ensure_ascii=False)

    print(f"\nVocabulary saved to: {output_file}")


# === Vocab Extraction =========

# AI
def clean_candidate(term):
    term = re.sub(r"\s+", " ", term).strip()
    term = term.strip(".,;:!?()[]{}\"'")

    return term


# AI
def extract_candidates(text):
    extractor = yake.KeywordExtractor(
        lan="en",
        n=MAX_NGRAM_SIZE,
        top=MAX_CANDIDATES,
        dedupLim=0.9,
    )

    keywords = extractor.extract_keywords(text)

    candidates = []
    seen = set()

    for term, score in keywords:
        term = clean_candidate(term)

        if len(term) < 2:
            continue

        if not re.search(r"[a-zA-Z]", term):
            continue

        normalized = term.casefold()

        if normalized in seen:
            continue

        seen.add(normalized)
        candidates.append(term)

    return candidates



def extract_vocab(m3av_dataset_dir, test_df):
    video_ids = test_df["video_id"].dropna().astype(str).unique()
    print(f"Unique videos in manifest: {len(video_ids)}")
    
    extracted_vocab = {i:[] for i in video_ids}
    
    for i, video_id in enumerate(video_ids):
        print(f"\n[{i}/{len(video_ids)}] Processing {video_id}")
        
        video_path = m3av_dataset_dir / video_id
        
        if not video_path.is_dir():
            raise FileNotFoundError(f"Video directory for {video_id} not found.")
        
        ocr_transcriptions = load_ocr_transcriptions(video_path)
        
        extracted_transcription = extract_candidates(ocr_transcriptions)
        if not extracted_transcription:
            raise ValueError(f"No vocabulary candidates extracted for {video_id}")
        
        extracted_vocab[video_id] = extracted_transcription
        
        print(f"Extracted candidates: {len(extracted_transcription)}")
    
    return extracted_vocab



def main():
    m3av_dataset_dir = M3AV_DATASET_DIR
    test_manifest = TEST_MANIFEST
    test_df = load_csv(test_manifest) #.head(5)
    vocabulary = extract_vocab(m3av_dataset_dir, test_df)
    save_vocab(vocabulary, OUTPUT_DIR)
    
    total_vocabulary = sum(len(voc) for voc in vocabulary.values())
    
    print("Extraction Complete!")
    print(f"Videos processed: {len(test_df)}")
    print(f"Total vocabulary: {total_vocabulary}")



if __name__=="__main__":
    main()