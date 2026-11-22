from pathlib import Path
import re
from collections import defaultdict, Counter
import csv
from normalize_text import normalize_text
from nltk.corpus import stopwords


# === GLOBALS ===
WORDS = 20
NGRAM = 2

REMOVE_STOPWORDS = False # True to remove stopwords, otherwise False

# --- NLTK setup ---
# nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

if REMOVE_STOPWORDS:
    OUTPUT_DIR = Path(f"outputs/comparisons/words{WORDS}_ngram{NGRAM}/without_stopwords") # comparisons_no_stopwords
else:
    OUTPUT_DIR = Path(f"outputs/comparisons/words{WORDS}_ngram{NGRAM}/with_stopwords") # comparisons


# --- global variables to be used ---

VARIANTS = {
    "baseline": Path("outputs/baseline_evaluation"),
    "training": Path("outputs/training_evaluation"),
    "baseline_word_boosting": Path(f"outputs/word_boosting_words{WORDS}_ngram{NGRAM}/baseline_word_boosting_evaluation"),
    "training_word_boosting": Path(f"outputs/word_boosting_words{WORDS}_ngram{NGRAM}/training_word_boosting_evaluation")
}

ORDER = ["baseline", "baseline_word_boosting", "training", "training_word_boosting"]

COMPARISONS = [
    ("baseline", "baseline_word_boosting"),
    ("training", "training_word_boosting"),
    ("baseline", "training"),
    ("baseline_word_boosting", "training_word_boosting"),
    ("baseline", "training_word_boosting"),
    ("training", "baseline_word_boosting"),
]

# --- _/\_/\_/\_/\_/\_/\_/\_/\_/\_ ---


# === BUILD FILES LIST ===

def build_files_list():
    files = {}
    
    for name, path in VARIANTS.items():
        paths = list(path.rglob("*.txt"))
        files[name] = paths
    
    return files


# === EXTRACT MODEL NAME ===

def get_model_name(name):
    name = name.stem
    name = name.split("_")
    model_name = name[:2]
    model_name = "_".join(model_name)
    return model_name


# === LOAD TEXT DATA ===

def load_data(files):
    """
    Load the text from the files.

    Args:
        files (dict): contains all filepaths for the 4 model conditions.

    Raises:
        ValueError: if duplicate model files exist.

    Returns:
        dict: text data in dict format. 
              {baseline: {qwen_0.6B: text, ...}, ...}
    """
    data = defaultdict(dict)
    for name, paths in files.items():
        for path in paths:
            model = get_model_name(path)
            if model in data[name]:
                raise ValueError(f"Duplicate model files: model = {model}, condition = {name}")
            data[name][model] = {
                "path": path,
                "text": path.read_text(encoding="utf-8")
            }
    return data


# === PARSE TEXT DATA ===
# AI
def parse_results(text):

    wer_match = re.search(r"WER:\s*([\d.]+)", text)

    if not wer_match:
        raise ValueError("Could not find overall WER.")

    overall_wer = float(wer_match.group(1))

    sentences = {}

    current_sentence = None
    pending_ref = None

    for line in text.splitlines():
        line = line.strip()
        sentence_match = re.match(
            r"===\s*SENTENCE\s+(\d+)\s*===",
            line
        )
        
        if sentence_match:
            current_sentence = int(sentence_match.group(1))
            sentences.setdefault(current_sentence, {
                "ref": [],
                "hyp": []
            })
            pending_ref = None
            continue

        if current_sentence is None:
            continue

        if line.startswith("REF:"):
            pending_ref = normalize_text(line[4:].replace("*", ""))

        elif line.startswith("HYP:") and pending_ref is not None:
            sentences[current_sentence]["ref"].append(pending_ref)
            sentences[current_sentence]["hyp"].append(normalize_text(line[4:].replace("*", "")))
            pending_ref = None

    parsed_sentences = {}

    for sentence_id, values in sentences.items():
        parsed_sentences[sentence_id] = {
            "ref": normalize_text(" ".join(values["ref"])),
            "hyp": normalize_text(" ".join(values["hyp"]))
        }

    return {"wer": overall_wer, "sentences": parsed_sentences}


# === ALIGN WORDS FROM TEXT ===

def align_words(ref, hyp):

    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()

    rows = len(ref_words)
    cols = len(hyp_words)

    dp = [[0] * (cols + 1) for _ in range(rows + 1)]

    for i in range(rows + 1):
        dp[i][0] = i

    for j in range(cols + 1):
        dp[0][j] = j

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = (0 if ref_words[i - 1] == hyp_words[j - 1] else 1)

            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Backtrack to identify individual errors.
    alignment = []
    i, j = rows, cols

    while i > 0 or j > 0:

        if (i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1] and dp[i][j] == dp[i - 1][j - 1]):
            alignment.append(("C", ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1

        elif (i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1):
            alignment.append(("S", ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1

        elif (j > 0 and dp[i][j] == dp[i][j - 1] + 1):
            alignment.append(("I", "", hyp_words[j - 1]))
            j -= 1

        else:
            alignment.append(("D", ref_words[i - 1], ""))
            i -= 1

    alignment.reverse()
    return alignment


# === CALCULATE WER ===

def calculate_wer_from_alignment(alignment, ref):

    errors = sum(1 for operation, _, _ in alignment if operation != "C")
    reference_words = len(ref.split())
    
    return errors / reference_words if reference_words else 0.0


# === GET ALL SENTENCE DATA ===

def get_sentence_data(model_results, sentence_id):
    available = {}
    for condition in ORDER:
        result = model_results[condition]
        if result is None:
            continue
        sentence = result["sentences"].get(sentence_id)
        if sentence is not None:
            available[condition] = sentence
    if not available:
        return None

    # # References should be identical across conditions.
    # references = {sentence["ref"] for sentence in available.values()}

    # if len(references) > 1:
    #     raise ValueError(f"Different references found for sentence {sentence_id}: {references}")
    
    references = {
        condition: sentence["ref"]
        for condition, sentence in available.items()
    }

    if len(set(references.values())) > 1:
        details = "\n".join(
            f"{condition}: {ref!r}"
            for condition, ref in references.items()
        )

        raise ValueError(
            f"Different references found for sentence {sentence_id}:\n{details}"
        )

    ref = next(iter(references.values()))
    sentence_results = {}
    
    for condition in ORDER:
        result = model_results[condition]
        if result is None:
            sentence_results[condition] = None
            continue

        sentence = result["sentences"].get(sentence_id)

        if sentence is None:
            hyp = ref
        else:
            hyp = sentence["hyp"]

        alignment = align_words(ref, hyp)

        sentence_results[condition] = {
            "ref": ref,
            "hyp": hyp,
            "alignment": alignment,
            "wer": calculate_wer_from_alignment(
                alignment, ref
            )
        }

    return sentence_results


# === PREPARE RESULTS ===

def prepare_results(data):
    results = {}

    all_models = sorted({
        model
        for condition in data.values()
        for model in condition
    })

    for model in all_models:
        results[model] = {}
        for condition in ORDER:
            if model not in data.get(condition, {}):
                results[model][condition] = None
                continue

            file_data = data[condition][model]

            parsed = parse_results(file_data["text"])

            results[model][condition] = {
                **parsed,
                "path": file_data["path"]
            }

    return results


# === CREATE COMPARISONS ===

def create_overall_comparison(results):
    rows = []
    for model, model_results in results.items():
        available = {
            condition: result
            for condition, result in model_results.items()
            if result is not None
        }

        wer_values = {condition: result["wer"] for condition, result in available.items()}

        baseline_wer = wer_values.get("baseline")
        training_wer = wer_values.get("training")

        row = {"model": model}

        for condition in ORDER:
            row[condition] = wer_values.get(condition)

        for condition in ["baseline_word_boosting", "training_word_boosting"]:
            original = (
                baseline_wer if condition == "baseline_word_boosting"
                else training_wer
            )

            boosted = wer_values.get(condition)

            if original is not None and boosted is not None:
                improvement = ( (original - boosted) / original * 100 if original != 0 else None )
                row[f"{condition}_improvement_percent"] = improvement
            else:
                row[f"{condition}_improvement_percent"] = None

        # Improvement from baseline to training
        for condition_a, condition_b, column_name in [
            ("baseline", "training", "training_improvement_percent"),
            ("baseline_word_boosting", "training_word_boosting", "training_word_boosting_improvement_percent")
        ]:
            original = wer_values.get(condition_a)
            improved = wer_values.get(condition_b)

            if original is not None and improved is not None:
                improvement = (
                    (original - improved) / original * 100
                    if original != 0 else None
                )
                row[column_name] = improvement
            else:
                row[column_name] = None
        
        rows.append(row)

    return rows


def create_pairwise_comparison(results):
    rows = []
    for model, model_results in results.items():
        for condition_a, condition_b in COMPARISONS:
            result_a = model_results[condition_a]
            result_b = model_results[condition_b]

            if result_a is None or result_b is None:
                continue

            wer_a = result_a["wer"]
            wer_b = result_b["wer"]

            rows.append({
                "model": model,
                "condition_a": condition_a,
                "file_a": str(result_a["path"]),
                "wer_a": wer_a,
                "condition_b": condition_b,
                "file_b": str(result_b["path"]),
                "wer_b": wer_b,
                "absolute_wer_change": wer_b - wer_a,
                "relative_improvement_percent": ( (wer_a - wer_b) / wer_a * 100 if wer_a != 0 else None )
            })

    return rows


def create_error_frequency(results):

    frequencies = defaultdict(Counter)

    for model, model_results in results.items():

        sentence_ids = set()

        for result in model_results.values():

            if result is not None:
                sentence_ids.update(result["sentences"])

        for sentence_id in sorted(sentence_ids):

            sentence_data = get_sentence_data(
                model_results,
                sentence_id
            )

            if sentence_data is None:
                continue

            for condition, sentence in sentence_data.items():

                if sentence is None:
                    continue

                for operation, ref_word, hyp_word in sentence["alignment"]:

                    if operation == "C":
                        continue
                    
                    if REMOVE_STOPWORDS:
                        if ref_word in STOPWORDS or hyp_word in STOPWORDS:
                            continue
                    
                    frequencies[
                        (model, operation, ref_word, hyp_word)
                    ][condition] += 1

    rows = []

    for (model, operation, ref_word, hyp_word), counts in frequencies.items():

        row = {
            "model": model,
            "error_type": operation,
            "reference_word": ref_word,
            "recognized_word": hyp_word
        }

        for condition in ORDER:
            row[condition] = counts[condition]

        rows.append(row)

    rows.sort(
        key=lambda row: sum(
            row[condition] or 0
            for condition in ORDER
        ),
        reverse=True
    )

    return rows


def create_persistent_errors(results):

    rows = defaultdict(list)

    for model, model_results in results.items():
        sentence_ids = set()
        
        for result in model_results.values():
            if result is not None:
                sentence_ids.update(result["sentences"])

        for sentence_id in sorted(sentence_ids):
            sentence_data = get_sentence_data(model_results, sentence_id)

            if sentence_data is None:
                continue

            errors_by_condition = {}

            for condition, sentence in sentence_data.items():
                if sentence is None:
                    continue
                if REMOVE_STOPWORDS:
                    errors_by_condition[condition] = Counter(
                        (operation, ref_word, hyp_word)
                        for operation, ref_word, hyp_word
                        in sentence["alignment"]
                        if operation != "C"
                        and ref_word not in STOPWORDS
                        and hyp_word not in STOPWORDS
                    )
                else:
                    errors_by_condition[condition] = Counter(
                        (operation, ref_word, hyp_word)
                        for operation, ref_word, hyp_word
                        in sentence["alignment"]
                        if operation != "C"
                    )

            if len(errors_by_condition) != 4:
                continue

            common_errors = set.intersection( *[set(errors) for errors in errors_by_condition.values()] )

            for error in sorted(common_errors):
                rows[model].append({
                    #"model": model,
                    "sentence": sentence_id,
                    "error_type": error[0],
                    "reference_word": error[1],
                    "recognized_word": error[2],
                    "frequency_in_each_condition": min(
                        errors[error]
                        for errors in errors_by_condition.values()
                    )
                })

    return rows


# === COMPARISONS ===

def compare(data):
    results = prepare_results(data)
    
    overall = create_overall_comparison(results)
    pairwise = create_pairwise_comparison(results)
    frequencies = create_error_frequency(results)
    persistent = create_persistent_errors(results)

    return {
        "results": results,
        "overall": overall,
        "pairwise": pairwise,
        "frequencies": frequencies,
        "persistent": persistent
    }


# === WRITE RESULTS ===

def write_csv(filename, rows):

    if not rows:
        print(f"WARNING: No data for {filename}")
        return

    output_path = OUTPUT_DIR / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")


def write_model_reports(results):

    for model, model_results in results.items():
        output_path = (OUTPUT_DIR / "reports" / f"{model}_report.txt")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        lines.append(f"MODEL: {model}")
        lines.append("=" * 80)
        lines.append("SOURCE FILES")

        for condition in ORDER:
            result = model_results[condition]
            if result is None:
                lines.append(f"{condition}: MISSING")
            else:
                lines.append(f"{condition}: {result['path']}")

        lines.append("\nOVERALL WER")
        lines.append("-" * 80)

        for condition in ORDER:
            result = model_results[condition]
            if result is not None:
                lines.append(f"{condition}: {result['wer']:.4f}")

        sentence_ids = sorted({
            sid
            for result in model_results.values()
            if result is not None
            for sid in result["sentences"]
        })

        lines.append("\nSENTENCE-LEVEL ERROR COMPARISON")
        lines.append("=" * 80)

        for sentence_id in sentence_ids:
            sentence_data = get_sentence_data(model_results, sentence_id)
            
            if sentence_data is None:
                continue
                
            lines.append(f"\nSENTENCE {sentence_id}")
            lines.append("-" * 80)

            ref = next(sentence["ref"] for sentence in sentence_data.values() if sentence is not None)

            lines.append(f"REF: {ref}")

            for condition in ORDER:
                sentence = sentence_data[condition]
                
                if sentence is None:
                    lines.append(f"{condition}: MISSING")
                    continue
                
                lines.append(
                    f"{condition} "
                    f"(WER: {sentence['wer']:.4f}): "
                    f"{sentence['hyp']}"
                )

                errors = [
                    (op, ref_word, hyp_word)
                    for op, ref_word, hyp_word
                    in sentence["alignment"]
                    if op != "C"
                    and (
                        not REMOVE_STOPWORDS
                        or (ref_word not in STOPWORDS and hyp_word not in STOPWORDS)
                    )
                ]

                if errors:
                    error_text = "; ".join(f"{op}: {ref_word} -> {hyp_word}" for op, ref_word, hyp_word in errors)
                    lines.append(f"Errors: {error_text}")
                else:
                    lines.append("Errors: None")

            # Compare exact errors across conditions.
            error_sets = {}

            for condition, sentence in sentence_data.items():
                if sentence is not None:
                    error_sets[condition] = Counter(
                        (op, ref_word, hyp_word)
                        for op, ref_word, hyp_word
                        in sentence["alignment"]
                        if op != "C"
                        and (
                            not REMOVE_STOPWORDS
                            or (ref_word not in STOPWORDS and hyp_word not in STOPWORDS)
                        )
                    )

            if len(error_sets) == 4:
                lines.append("\nERROR CHANGES")
                for condition_a, condition_b in COMPARISONS:
                    errors_a = error_sets[condition_a]
                    errors_b = error_sets[condition_b]

                    corrected = errors_a - errors_b
                    introduced = errors_b - errors_a

                    lines.append(f"\n{condition_a} -> {condition_b}")
                    lines.append(f"Corrected errors: {list(corrected.elements())}")
                    lines.append(f"Introduced errors: {list(introduced.elements())}")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        
        print(f"Saved: {output_path}")
        

def write_result(result):
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    write_csv("overall_comparison.csv", result["overall"])
    write_csv("pairwise_comparison.csv", result["pairwise"])
    write_csv("error_frequency.csv", result["frequencies"])
    
    # write_csv("persistent_errors.csv", result["persistent"])
    persistent_dir = OUTPUT_DIR / "persistent_errors"
    persistent_dir.mkdir(parents=True, exist_ok=True)
    for model, errors in result["persistent"].items():
        write_csv(
            f"persistent_errors/{model}.csv",
            errors
        )
    
    write_model_reports(result["results"])


def main():
    # build file list
    all_files = build_files_list()
    print("\nFiles extracted...\n")
    
    # Load data
    data = load_data(all_files)
    print("Files loaded...\n")
    
    # compare data
    result = compare(data)
    
    # write results
    write_result(result)


if __name__=="__main__":
    main()