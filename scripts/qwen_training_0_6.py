from pathlib import Path
import re
import time
import math
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import torch

from transformers import (AutoProcessor, Qwen3ASRForConditionalGeneration)

# === Constants ===

SIZE = "0.6B" # set size to "0.6B" or "1.7B"
TRAIN_MANIFEST_CSV = Path("data/manifests/m3av_train_manifest.csv")
DEV_MANIFEST_CSV = Path("data/manifests/m3av_dev_manifest.csv")
DATA_DIR = Path("data/processed/m3av_dataset/segments")

QWEN_MODEL_ID = f"Qwen/Qwen3-ASR-{SIZE}-hf" # Qwen/Qwen3-ASR-1.7B-hf or Qwen/Qwen3-ASR-0.6B-hf
MODELS_DIR = Path(f"outputs/training/qwen_{SIZE}")

TRAIN_SUBSET_FRACTION = 0.00005
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1

SEED = 42

# AI CONST suggestions
MAX_NEW_TOKENS = 256

USE_MIXED_PRECISION = True


# === Helper Functions ===

def normalize_text(text):
    text = text.lower()
    text = text.replace("'", "")                # remove apostrophes without splitting contractions
    text = re.sub(r"[-‐‑‒–—]", " ", text)       # treat hyphens and dashes as word separators
    text = re.sub(r"[^\w\s]", " ", text)        # replace remaining punctuation with spaces
    text = re.sub(r"\s+", " ", text).strip()    # normalize whitespace
    return text


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


def check_files(data, split):
    audio_files = [data[i]['audio'] for i in range(len(data))]
    missing_files = [str(path) for path in audio_files if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} audio files in the {split} dataset are missing. "
            f"First examples: {missing_files[:5]}"
        )
    print(f"All {split} files exist at the correct path.")



# === Model =====================

def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. For this full fine-tuning script, use a suitable GPU node.")
    
    device = torch.device("cuda")
    
    print(f"Using device: {device}")
    print(f"Model ID: {QWEN_MODEL_ID}")
    
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(QWEN_MODEL_ID)
    
    model.to(device)

    # AI
    # Disable KV caching during training to avoid unnecessary
    # memory use. Restore it for generation during validation.
    model.config.use_cache = False

    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    return model, processor, device



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
    
    if len(audio_files)!=len(transcriptions):
        raise ValueError("Number of audio files is different than number of transcriptions.")
    
    data = []
    for i in range(len(audio_files)):
        data.append({
            "audio": audio_files[i], 
            "text": f"language English<asr_text>{transcriptions[i]}", 
            "reference": normalize_text(transcriptions[i]), 
            "sample_id": str(df["sample_id"].iloc[i])
        })
    
    return data



# === AI suggested helper functions ===

def get_amp_settings(device):
    if (
        USE_MIXED_PRECISION
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        return True, torch.bfloat16, False

    if USE_MIXED_PRECISION and device.type == "cuda":
        return True, torch.float16, True

    return False, torch.float32, False


def autocast_context(device, enabled, dtype):
    if enabled:
        return torch.autocast(
            device_type=device.type,
            dtype=dtype,
        )

    return nullcontext()


# === Qwen input preparation === AI used this: https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf#training--fine-tuning

def make_training_conversations(batch):
    """
    Create one conversation per example.

    The processor reads each local audio file and builds the
    model-specific audio and text inputs.
    """

    conversations = []

    for item in batch:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": str(item["audio"]),
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": item["text"],
                    }
                ],
            },
        ]

        conversations.append(conversation)

    return conversations


def prepare_training_batch(batch, processor, device):
    conversations = make_training_conversations(batch)

    inputs = processor.apply_chat_template(
        conversations, tokenize=True, return_dict=True, 
        processor_kwargs={"output_labels": True}
    )

    return inputs.to(device)


def prepare_generation_batch(batch, processor, device):
    audio_paths = [
        str(item["audio"]) for item in batch
    ]

    inputs = processor.apply_transcription_request(
        audio=audio_paths,
        language=["English"] * len(audio_paths),
    )

    return inputs.to(device)


# === Validation ================

def validate_qwen(model, processor, dev_loader, device, amp_enabled, amp_dtype):
    model.eval()
    
    total_loss = 0.0
    total_valid_tokens = 0

    total_word_errors = 0
    total_reference_words = 0
    
    for batch in dev_loader:

        # ----------------------------------------------------
        # Teacher-forced validation loss
        # ----------------------------------------------------

        training_inputs = prepare_training_batch(batch, processor, device)

        labels = training_inputs.get("labels")

        if labels is None:
            raise RuntimeError(
                "The processor did not produce training labels. "
                "Check your Transformers version and Qwen training API."
            )

        valid_tokens = (labels != -100).sum().item()

        if valid_tokens == 0:
            raise ValueError("Validation batch contains no valid target tokens.")

        with torch.no_grad():
            with autocast_context(device, amp_enabled, amp_dtype):
                outputs = model(**training_inputs)

        if outputs.loss is None:
            raise RuntimeError("Qwen returned no training loss.")

        # The model loss is averaged over valid target tokens.
        total_loss += outputs.loss.float().item() * valid_tokens
        total_valid_tokens += valid_tokens

        # ----------------------------------------------------
        # Actual transcription and WER
        # ----------------------------------------------------

        generation_inputs = prepare_generation_batch(batch, processor, device)

        # Cache can be enabled for generation.
        model.config.use_cache = True
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = True

        with torch.no_grad():
            with autocast_context(device, amp_enabled, amp_dtype):
                output_ids = model.generate(
                    **generation_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False

        generated_ids = output_ids[:, generation_inputs["input_ids"].shape[1]:]

        hypotheses = processor.decode(generated_ids, return_format="transcription_only",)

        for item, hypothesis in zip(batch, hypotheses):
            reference = normalize_text(item["reference"])
            hypothesis = normalize_text(hypothesis)

            errors, reference_words = word_edit_distance(reference, hypothesis)

            total_word_errors += errors
            total_reference_words += reference_words

    if total_valid_tokens == 0:
        raise ValueError("No valid validation tokens.")

    if total_reference_words == 0:
        raise ValueError("Validation references contain no words.")

    mean_loss = total_loss / total_valid_tokens
    wer = total_word_errors / total_reference_words

    return mean_loss, wer



# === Plotting ================== AI end after plotting

def save_loss_plot(train_batch_losses, train_epoch_losses, dev_losses, output_dir, epoch):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Training loss after each batch
    axes[0].plot(range(1, len(train_batch_losses) + 1), train_batch_losses, alpha=0.7, label="Batch loss")
    axes[0].set_xlabel("Training batch")
    axes[0].set_ylabel("Loss per valid token")
    axes[0].set_title("Training Loss per Batch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Epoch-level training and development losses
    epochs = range(1, len(train_epoch_losses) + 1)

    axes[1].plot(epochs, train_epoch_losses, marker="o", label="Training loss",)
    axes[1].plot(epochs, dev_losses, marker="o", label="Development loss",)
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


# === Save checkpoint ===

def save_checkpoint_atomic(model, processor, output_dir):
    checkpoint_dir = output_dir / f"latest_saved_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    print(f"Temp Checkpoint saved at: {checkpoint_dir}", flush=True)
    

# === Training ================== (AI mixed)

def train_qwen(model, processor, train_dataset, dev_dataset, output_dir, device):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # AI changes to original code
    amp_enabled, amp_dtype, use_grad_scaler = (
        get_amp_settings(device)
    )
    
    print(f"Mixed precision: {amp_enabled}")
    if amp_enabled:
        print(f"Autocast dtype: {amp_dtype}")
    
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    
    # why not use the collate_fn function from whisper?
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: batch)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch:batch)
    
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
    
    # why adamW, how is it different?
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
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

        for batch_idx, batch in enumerate(train_loader):
            # Stop after running on train (sub-)set
            if batch_idx >= num_batches:
                break
            
            inputs = prepare_training_batch(batch, processor, device)
            
            labels = inputs.get("labels")
            
            if labels is None:
                raise RuntimeError(
                    "The processor did not produce labels. "
                    "Cannot calculate supervised training loss."
                )

            valid_tokens = (labels != -100).sum().item()

            if valid_tokens == 0:
                raise ValueError("Training batch contains no valid target tokens.")

            with autocast_context(device, amp_enabled, amp_dtype):
                outputs = model(**inputs)

            if outputs.loss is None:
                raise RuntimeError("Qwen returned no training loss.")
            
            # Convert mean token loss into summed token loss.
            # This permits token-weighted gradient accumulation.
            loss_sum = outputs.loss.float() * valid_tokens

            if use_grad_scaler:
                scaler.scale(loss_sum).backward()
            else:
                loss_sum.backward()

            accumulated_valid_tokens += valid_tokens
            total_valid_tokens += valid_tokens
            total_loss += loss_sum.detach().item()
            
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
                if use_grad_scaler:
                    scaler.unscale_(optimizer)
                
                # Normalize by the number of valid tokens in
                # this entire accumulation window.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_valid_tokens)
                
                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                
                # Save a checkpoint every 104 training batches,
                # after a completed optimizer update.
                if (batch_idx + 1) % 104 == 0:
                    save_checkpoint_atomic(model, processor, output_dir)
                accumulated_valid_tokens = 0
        
        if total_valid_tokens == 0:
            raise ValueError("No training tokens were processed.")

        epoch_train_loss = total_loss / total_valid_tokens
        
        # Save the model and processor together so the checkpoint
        # can be loaded again using Hugging Face Transformers.
        checkpoint_dir = output_dir / f"epoch_{epoch + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)

        print(f"Checkpoint saved to: {checkpoint_dir}")

        dev_loss, dev_wer = validate_qwen(
            model,
            processor,
            dev_loader,
            device,
            amp_enabled,
            amp_dtype,
        )

        train_losses.append(epoch_train_loss)
        dev_losses.append(dev_loss)
        dev_wers.append(dev_wer)

        print(f"\nEpoch {epoch + 1} results:")
        print(f"  Training loss: {epoch_train_loss:.4f}")
        print(f"  Development loss: {dev_loss:.4f}")
        print(f"  Development WER: {dev_wer:.2%}")

        save_loss_plot(
            train_batch_losses,
            train_losses,
            dev_losses,
            output_dir,
            epoch,
        )

        metrics_path = (
            output_dir / f"training_metrics_{epoch + 1}.csv"
        )

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
    
    torch.manual_seed(SEED)
    
    output_dir = MODELS_DIR / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output dir: {output_dir}")
    print("\nLoading Model...")
    model, processor, device = load_model()
    
    print("\nLoading Manifests...")
    train_data = load_manifest(TRAIN_MANIFEST_CSV, "train")
    dev_data = load_manifest(DEV_MANIFEST_CSV, "dev")
    check_files(train_data, "train")
    check_files(dev_data, "dev")
    
    print("\nStarting Qwen fine-tuning and validation...")
    train_qwen(model, processor, train_data, dev_data, output_dir, device)
    
    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed / 60:.2f} minutes.")
    print(f"Results saved under: {output_dir}")



# === Call Main ===

if __name__=="__main__":
    main()