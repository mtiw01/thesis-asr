import re

def normalize_text(text):
    text = text.lower()
    text = text.replace("'", "")                # remove apostrophes without splitting contractions
    text = re.sub(r"[-‐‑‒–—]", " ", text)       # treat hyphens and dashes as word separators
    text = re.sub(r"[^\w\s]", " ", text)        # replace remaining punctuation with spaces
    text = re.sub(r"\s+", " ", text).strip()    # normalize whitespace
    return text