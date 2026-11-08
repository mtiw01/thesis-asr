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
import pandas as pd

from helpers_whisper_training import LabelSmoothingLoss

from whisper_training import (
    check_files, 
    collate_fn, 
    load_manifest, 
    AudioDataset, 
    prepare_tokens, 
    validate_whisper
)

# === Constants ===

SIZE = "turbo" # set size to "medium" or "turbo"
TRAIN_MANIFEST_CSV = Path("data/manifests/m3av_train_manifest.csv")
DEV_MANIFEST_CSV = Path("data/manifests/m3av_dev_manifest.csv")
DATA_DIR = Path("data/processed/m3av_dataset/segments")

MODELS_DIR = Path(f"outputs/training/whisper_{SIZE}")

TRAIN_SUBSET_FRACTION = 1.0
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1

LABEL_SMOOTHING = 0.1
SEED = 42


# === Model =====================

def load_latest_checkpoint(size, models_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Find completed epoch checkpoints.
    checkpoint_files = []

    for path in models_dir.glob("epoch*.pt"):
        match = re.fullmatch(r"epoch(\d+)\.pt", path.name)
        if match:
            checkpoint_files.append((int(match.group(1)), path))

    if not checkpoint_files:
        raise FileNotFoundError(f"No epoch checkpoints found in {models_dir}")

    latest_epoch, checkpoint_path = max(checkpoint_files, key=lambda item: item[0])

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load the complete checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True,)

    # Check that the checkpoint belongs to the requested model.
    if checkpoint["model_size"] != size:
        raise ValueError(f"Checkpoint is for {checkpoint['model_size']}, but requested model is {size}.")

    if checkpoint["batch_size"] != BATCH_SIZE:
        raise ValueError("Batch size differs from the saved checkpoint.")

    if checkpoint["grad_accum_steps"] != GRAD_ACCUM_STEPS:
        raise ValueError("Gradient accumulation differs from the saved checkpoint.")

    if checkpoint["learning_rate"] != LEARNING_RATE:
        raise ValueError("Learning rate differs from the saved checkpoint.")

    # Load the Whisper architecture.
    model = whisper.load_model(size, device=device)

    # Restore model weights.
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model: {size}, mel bins: {model.dims.n_mels}")
    print(f"Continuing from epoch {latest_epoch}")

    return model, checkpoint, latest_epoch, device


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
    epochs = range(epoch - len(train_epoch_losses) + 1, epoch + 1)

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

    plot_path = output_dir / f"loss_{epoch}.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"Loss plot saved to: {plot_path}")



# Temporary checkpoint
def save_checkpoint_atomic(model, optimizer, checkpoint_path, epoch):
    """Save model and optimizer states."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_size": SIZE,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
    }

    temp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}", flush=True)



# === Training ==================

def train_whisper(model, train_dataset, dev_dataset, output_dir, checkpoint, starting_epoch):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=checkpoint["learning_rate"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    decoder_criterion = LabelSmoothingLoss(ignore_index=-1, label_smoothing=LABEL_SMOOTHING, reduction="sum")
    
    train_losses = []
    dev_losses = []
    dev_wers = []
    
    train_batch_losses = []
    
    for epoch in range(NUM_EPOCHS):
        global_epoch = starting_epoch + epoch + 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_valid_tokens = 0
        total_loss = 0.0
        total_valid_tokens = 0
        
        print(f"\n=== Epoch {global_epoch} ===")

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
                    save_checkpoint_atomic(model, optimizer, output_dir / "completed_batch.pt", global_epoch)	

                accumulated_valid_tokens = 0
        
        if total_valid_tokens == 0:
            raise ValueError("No training tokens were processed.")

        epoch_train_loss = total_loss / total_valid_tokens

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

        # Save model weights after training completes for this epoch.
        checkpoint_path = output_dir / f"epoch{global_epoch}.pt"
        
        if checkpoint_path.exists():
            raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")

        save_checkpoint_atomic(model, optimizer, checkpoint_path, global_epoch)
        print(f"Model weights saved to: {checkpoint_path}")


        train_losses.append(epoch_train_loss)
        dev_losses.append(dev_loss)
        dev_wers.append(dev_wer)

        print(f"\nEpoch {global_epoch} results:")
        print(f"  Training loss: {epoch_train_loss:.4f}")
        print(f"  Development loss: {dev_loss:.4f}")
        print(f"  Development WER: {dev_wer:.2%}")

        # Keep the loss plot updated after every epoch.
        save_loss_plot(train_batch_losses, train_losses, dev_losses, output_dir, global_epoch)

        # Save metrics so they can be analyzed later.
        metrics_path = output_dir / f"training_metrics_{global_epoch}.csv"

        pd.DataFrame({
            "epoch": [global_epoch],
            "train_loss": [epoch_train_loss],
            "dev_loss": [dev_loss],
            "dev_wer": [dev_wer],
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
    model, checkpoint, starting_epoch, device = load_latest_checkpoint(SIZE, MODELS_DIR,)
    
    print("\nLoading Manifests...")
    train_audio_files, train_transcriptions = load_manifest(TRAIN_MANIFEST_CSV, "train")
    dev_audio_files, dev_transcriptions = load_manifest(DEV_MANIFEST_CSV, "dev")
    check_files(train_audio_files, "train")
    check_files(dev_audio_files, "dev")
    
    print("\nCreating dataset...")
    train_dataset = AudioDataset(train_audio_files, train_transcriptions, n_mels=model.dims.n_mels)
    dev_dataset = AudioDataset(dev_audio_files, dev_transcriptions, n_mels=model.dims.n_mels)
    
    print("\nStart of Finetuning and Validation...")
    train_whisper(model, train_dataset, dev_dataset, output_dir, checkpoint, starting_epoch)
    
    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed / 60:.2f} minutes.")
    print(f"Results saved under: {output_dir}")



# === Call Main ===

if __name__=="__main__":
    main()
