from pathlib import Path
import re
import time
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import whisper
import whisper.audio as whisper_audio
import pandas as pd

from helpers_whisper_training import LabelSmoothingLoss, batch_tensors

# === Constants ===

SIZE = "medium" # set size to "medium" or "turbo"
TRAIN_MANIFEST_CSV = Path("data/manifests/m3av_train_manifest.csv")
DEV_MANIFEST_CSV = Path("data/manifests/m3av_dev_manifest.csv")
DATA_DIR = Path("data/processed/m3av_dataset/segments")

WHISPER_DIR = Path(f"outputs/training/whisper_{SIZE}")
MODELS_DIR = Path(f"outputs/training/whisper_{SIZE}")

TRAIN_SUBSET_FRACTION = 1.0
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1

LABEL_SMOOTHING = 0.1
SEED = 42



# === Helper Functions ===

def normalize_text(text):
    text = text.lower()
    text = text.replace("'", "")                # remove apostrophes without splitting contractions
    text = re.sub(r"[-‐‑‒–—]", " ", text)       # treat hyphens and dashes as word separators
    text = re.sub(r"[^\w\s]", " ", text)        # replace remaining punctuation with spaces
    text = re.sub(r"\s+", " ", text).strip()    # normalize whitespace
    return text


def check_files(audio_files, split):
    missing_files = [str(path) for path in audio_files if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} audio files in the {split} dataset are missing. "
            f"First examples: {missing_files[:5]}"
        )
    else:
        print(f"All {split} files exist at the correct path.")


def extract_audio_features(file, n_mels):
    audio = whisper_audio.load_audio(str(file))
    n_samples = len(audio)
    if n_samples > whisper_audio.N_SAMPLES:
        raise ValueError(
            f"Audio exceeds Whisper's 30-second input window: "
            f"{file} ({n_samples / whisper_audio.SAMPLE_RATE:.2f}s)"
        )
    padding_length = whisper_audio.N_SAMPLES - n_samples
    mel = whisper_audio.log_mel_spectrogram(audio, n_mels=n_mels, padding=padding_length,)
    return mel


def collate_fn(batch):
    mels, transcriptions = zip(*batch)
    return torch.stack(mels), transcriptions



# === Model =====================

def load_model(size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = whisper.load_model(size, device=torch.device(device))
    print(f"Model: {SIZE}, mel bins: {model.dims.n_mels}")
    return model



# === Manifest ==================

def load_manifest(file, split):
    df = pd.read_csv(file)
    required_columns = {"sample_id", "reference"}
    
    missing = required_columns - set(df.columns)
    if missing: 
        raise ValueError(f"There are missing columns in manifest.\nMissing Columns: {missing}")
    if df["sample_id"].isna().any():
        raise ValueError("Manifest contains missing sample IDs.")
    if df["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample IDs.")
    if df["reference"].isna().any():
        raise ValueError("Manifest contains missing reference transcripts.")
    if df.empty:
        raise ValueError(f"The {split} manifest is empty.")
    
    audio_files = [ DATA_DIR / split /f"{samp_id}.wav" for samp_id in df["sample_id"].tolist()]
    transcriptions = df["reference"].apply(normalize_text).tolist()
    
    return audio_files, transcriptions



# === Dataset Handling ==========

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, transcriptions, n_mels):
        if len(audio_files) != len(transcriptions):
            raise ValueError(
                "Audio files and transcripts have different lengths."
            )

        self.audio_files = audio_files
        self.transcriptions = transcriptions
        self.n_mels = n_mels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        mel = extract_audio_features(self.audio_files[idx], n_mels=self.n_mels)
        return mel, self.transcriptions[idx]



# === Token Preparation ========= AI start

def prepare_tokens(texts, tokenizer, tokens_to_prepend, device):
    text_tokens_list = [
        tokens_to_prepend
        + tokenizer.encode(text)
        + [tokenizer.eot]
        for text in texts
    ]

    text_tokens_list = [
        torch.tensor(tokens, dtype=torch.long)
        for tokens in text_tokens_list
    ]

    # Decoder inputs: pad with EOT.
    prev_outputs_tokens = batch_tensors(
        [tokens[:-1] for tokens in text_tokens_list],
        pad_value=tokenizer.eot,
    ).to(device)

    # Targets: use -1 so the loss ignores padded positions.
    target_tokens = batch_tensors(
        [tokens[1:] for tokens in text_tokens_list],
        pad_value=-1,
    ).to(device)

    # Ignore the language/task/no-timestamps prefix targets.
    ignore_prefix_size = 3

    prev_outputs_tokens = prev_outputs_tokens
    target_tokens = target_tokens[:, ignore_prefix_size:]

    return prev_outputs_tokens, target_tokens, ignore_prefix_size



# === WER =======================

def word_edit_distance(reference, hypothesis):
    """Return the minimum word-level edit distance."""

    ref = reference.split()
    hyp = hypothesis.split()

    previous = list(range(len(hyp) + 1))

    for i, ref_word in enumerate(ref, start=1):
        current = [i]

        for j, hyp_word in enumerate(hyp, start=1):
            substitution = (
                previous[j - 1] + (ref_word != hyp_word)
            )
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1

            current.append(
                min(substitution, deletion, insertion)
            )

        previous = current

    return previous[-1], len(ref)



# === Validation ================

def validate_whisper(
    model,
    dev_loader,
    tokenizer,
    tokens_to_prepend,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_valid_tokens = 0

    total_word_errors = 0
    total_reference_words = 0

    # Keep decoding settings consistent with your baseline
    # if you want a directly comparable WER.
    decoding_options = whisper.DecodingOptions(
        language="en",
        task="transcribe",
        without_timestamps=True,
        fp16=False,
    )

    with torch.no_grad():
        for feats, texts in dev_loader:
            feats = feats.to(device)

            prev_outputs_tokens, target_tokens, ignore_prefix_size = (
                prepare_tokens(
                    texts,
                    tokenizer,
                    tokens_to_prepend,
                    device,
                )
            )

            encoder_out = model.encoder(feats)

            logits = model.decoder(
                prev_outputs_tokens,
                encoder_out,
            )

            logits = logits[:, ignore_prefix_size:, :]

            loss_sum = criterion(logits, target_tokens)

            valid_tokens = (
                target_tokens != -1
            ).sum().item()

            total_loss += loss_sum.item()
            total_valid_tokens += valid_tokens

            # Generate actual transcriptions for WER.
            # Decoding separately re-encodes the audio.
            results = whisper.decode(
                model,
                feats,
                decoding_options,
            )

            for reference, result in zip(texts, results):
                reference = normalize_text(reference)
                hypothesis = normalize_text(result.text)

                errors, reference_words = word_edit_distance(
                    reference,
                    hypothesis,
                )

                total_word_errors += errors
                total_reference_words += reference_words

    if total_valid_tokens == 0:
        raise ValueError(
            "Validation produced no valid target tokens."
        )

    if total_reference_words == 0:
        raise ValueError(
            "Validation references contain no words."
        )

    mean_loss = total_loss / total_valid_tokens
    wer = total_word_errors / total_reference_words

    return mean_loss, wer



# === Plotting ================== AI end after plotting

def save_loss_plot(
    train_batch_losses,
    train_epoch_losses,
    dev_losses,
    output_dir,
    epoch
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Training loss after each batch
    axes[0].plot(
        range(1, len(train_batch_losses) + 1),
        train_batch_losses,
        alpha=0.7,
        label="Batch loss",
    )
    axes[0].set_xlabel("Training batch")
    axes[0].set_ylabel("Loss per valid token")
    axes[0].set_title("Training Loss per Batch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Epoch-level training and development losses
    epochs = range(1, len(train_epoch_losses) + 1)

    axes[1].plot(
        epochs, train_epoch_losses,
        marker="o", label="Training loss",
    )
    axes[1].plot(
        epochs, dev_losses,
        marker="o", label="Development loss",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss per valid token")
    axes[1].set_title("Epoch-Level Loss")
    axes[1].set_xticks(list(epochs))
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()

    plot_path = output_dir / f"loss_{epoch+1}.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"Loss plot saved to: {plot_path}")



# Temporary checkpoint
def save_checkpoint_atomic(model, checkpoint_path):
    """Save model weights safely, replacing the target only after saving completes."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")

    torch.save(model.state_dict(), temp_path)
    os.replace(temp_path, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}", flush=True)



# === Training ==================

def train_whisper(model, train_dataset, dev_dataset, output_dir):
    torch.manual_seed(SEED)
    
    tokenizer = whisper.tokenizer.get_tokenizer(
            True,
            num_languages=model.num_languages,
            language="en",
            task="transcribe",
        )
    
    tokens_to_prepend = tokenizer.encode(
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        allowed_special={"<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"})
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    if len(train_loader) == 0:
        raise ValueError("Training dataset is empty.")

    if len(dev_loader) == 0:
        raise ValueError("Development dataset is empty.")
    
    num_batches = min(len(train_loader), max(1, math.ceil(len(train_loader) * TRAIN_SUBSET_FRACTION)))
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Development examples: {len(dev_dataset)}")
    print(f"Available training batches: {len(train_loader)}")
    print(f"Training batches per epoch: {num_batches}")
    print(f"Epochs: {NUM_EPOCHS}")
    
    # Set the device to GPU if available. 
    # redundant because of load_model but makes function reusable.
    device = next(model.parameters()).device
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    decoder_criterion = LabelSmoothingLoss(ignore_index=-1, label_smoothing=LABEL_SMOOTHING, reduction="sum")
    
    train_losses = []
    dev_losses = []
    dev_wers = []
    
    train_batch_losses = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_valid_tokens = 0
        total_loss = 0.0
        total_valid_tokens = 0
        
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")

        for batch_idx, (feats, texts) in enumerate(train_loader):
            # Stop after running on train (sub-)set
            if batch_idx >= num_batches:
                break
            
            # Pad sequences
            prev_outputs_tokens, target_tokens, ignore_prefix_size = prepare_tokens(texts, tokenizer, tokens_to_prepend, device)

            feats = feats.to(device)
            
            # Forward pass
            encoder_out = model.encoder(feats)
            text_logits = model.decoder(prev_outputs_tokens, encoder_out)
            text_logits = text_logits[:, ignore_prefix_size:, :]
            loss_sum = decoder_criterion(text_logits, target_tokens)
            
            valid_tokens = (target_tokens!=-1).sum().item()
            
            if valid_tokens==0:
                raise ValueError("Batch contains no valid target tokens.")
            
            # Accumulate gradients of summed token losses
            loss_sum.backward()

            accumulated_valid_tokens += valid_tokens
            total_valid_tokens += valid_tokens
            total_loss += loss_sum.item()

            batch_loss = loss_sum.item() / valid_tokens
            
            train_batch_losses.append(batch_loss)

            print(
                f"Batch {batch_idx + 1}/{num_batches} | "
                f"Token loss: {batch_loss:.4f}"
            )

            is_accumulation_boundary = (
                (batch_idx + 1) % GRAD_ACCUM_STEPS == 0
                or batch_idx + 1 == num_batches
            )

            if is_accumulation_boundary:
                # Normalize by the number of valid tokens in
                # this entire accumulation window.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(
                            accumulated_valid_tokens
                        )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Save a checkpoint every 104 training batches,
                # after a completed optimizer update.
                if (batch_idx + 1) % 104 == 0:
                    save_checkpoint_atomic(model, output_dir / "latest.pt")	

                accumulated_valid_tokens = 0
        
        if total_valid_tokens == 0:
            raise ValueError("No training tokens were processed.")

        epoch_train_loss = total_loss / total_valid_tokens

        # Save model weights after training completes for this epoch.
        checkpoint_path = output_dir / f"epoch{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model weights saved to: {checkpoint_path}")

        if TRAIN_SUBSET_FRACTION < 1.0:
            print(f"Smoke test complete ({TRAIN_SUBSET_FRACTION*100}% of train set); skipping full validation.")
            return

        dev_loss, dev_wer = validate_whisper(
            model,
            dev_loader,
            tokenizer,
            tokens_to_prepend,
            decoder_criterion,
            device,
        )

        train_losses.append(epoch_train_loss)
        dev_losses.append(dev_loss)
        dev_wers.append(dev_wer)

        print(f"\nEpoch {epoch + 1} results:")
        print(f"  Training loss: {epoch_train_loss:.4f}")
        print(f"  Development loss: {dev_loss:.4f}")
        print(f"  Development WER: {dev_wer:.2%}")

        # Keep the loss plot updated after every epoch.
        save_loss_plot(train_batch_losses, train_losses, dev_losses, output_dir, epoch)

        # Save metrics so they can be analyzed later.
        metrics_path = output_dir / f"training_metrics_{epoch+1}.csv"

        pd.DataFrame({
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "dev_loss": dev_losses,
            "dev_wer": dev_wers,
        }).to_csv(metrics_path, index=False)

        print(f"Metrics saved to: {metrics_path}")
    
    return model




# === Main Function Calls ============

def main():
    start_time = time.time()
    
    output_dir = MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output dir: {output_dir}")
    print("\nLoading Model...")
    model = load_model(SIZE)
    
    print("\nLoading Manifests...")
    train_audio_files, train_transcriptions = load_manifest(TRAIN_MANIFEST_CSV, "train")
    dev_audio_files, dev_transcriptions = load_manifest(DEV_MANIFEST_CSV, "dev")
    check_files(train_audio_files, "train")
    check_files(dev_audio_files, "dev")
    
    print("\nCreating dataset...")
    train_dataset = AudioDataset(train_audio_files, train_transcriptions, n_mels=model.dims.n_mels)
    dev_dataset = AudioDataset(dev_audio_files, dev_transcriptions, n_mels=model.dims.n_mels)
    
    print("\nStart of Finetuning and Validation...")
    train_whisper(model, train_dataset, dev_dataset, output_dir)
    
    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed / 60:.2f} minutes.")
    print(f"Results saved under: {output_dir}")



# === Call Main ===

if __name__=="__main__":
    main()
