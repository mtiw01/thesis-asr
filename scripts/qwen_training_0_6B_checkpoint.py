from pathlib import Path
import re
import time
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import torch

from transformers import (AutoProcessor, Qwen3ASRForConditionalGeneration)

from qwen_training_0_6B import (
    check_files, 
    load_manifest, 
    get_amp_settings, 
    autocast_context, 
    prepare_training_batch, 
    validate_qwen, 
    save_checkpoint_atomic
)


# === Constants ===

SIZE = "0.6B" # set size to "0.6B" or "1.7B"
TRAIN_MANIFEST_CSV = Path("data/manifests/m3av_train_manifest.csv")
DEV_MANIFEST_CSV = Path("data/manifests/m3av_dev_manifest.csv")

MODELS_DIR = Path(f"outputs/training/qwen_{SIZE}")

TRAIN_SUBSET_FRACTION = 0.00001
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1

SEED = 42

# AI CONST suggestions
MAX_NEW_TOKENS = 256

USE_MIXED_PRECISION = True



# === Model =====================


def load_latest_checkpoint():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. For this full fine-tuning script, use a suitable GPU node.")
    
    device = torch.device("cuda")
    
    epoch_checkpoints = []

    for path in MODELS_DIR.glob("epoch_*"):
        match = re.fullmatch(r"epoch_(\d+)", path.name)

        if match and path.is_dir():
            epoch_checkpoints.append((int(match.group(1)), path))

    if not epoch_checkpoints:
        raise FileNotFoundError(
            f"No epoch checkpoint found in {MODELS_DIR}. This script requires an existing epoch checkpoint."
        )

    # Select checkpoint with highest epoch number.
    latest_epoch, checkpoint_dir = max(
        epoch_checkpoints,
        key=lambda item: item[0]
    )    
    
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_dir}")
    print(f"Starting from epoch {latest_epoch + 1}.")
    
    processor = AutoProcessor.from_pretrained(checkpoint_dir)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(checkpoint_dir)
    
    model.to(device)

    model.config.use_cache = False

    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    return model, processor, device, latest_epoch



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
    epochs = range(epoch - len(train_epoch_losses) + 1, epoch + 1)

    axes[1].plot(epochs, train_epoch_losses, marker="o", label="Training loss",)
    axes[1].plot(epochs, dev_losses, marker="o", label="Development loss",)
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


# === Save checkpoint ===

def save_checkpoint(model, processor, output_dir, epoch_number):
    # Save the model and processor together so the checkpoint
    # can be loaded again using Hugging Face Transformers.
    
    checkpoint_dir = output_dir / f"epoch_{epoch_number}"
    
    if checkpoint_dir.exists():
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_dir}. "
            "Choose a different output directory or starting point."
        )
    
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)

    print(f"Checkpoint saved to: {checkpoint_dir}")
    
    return checkpoint_dir
    

# === Training ================== (AI mixed)

def train_qwen(model, processor, train_dataset, dev_dataset, output_dir, device, starting_epoch):
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
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

        for batch_idx, batch in enumerate(train_loader):
            # Stop after running on train (sub-)set
            if batch_idx >= num_batches:
                break
            
            inputs = prepare_training_batch(batch, processor, device)
            
            labels = inputs.get("labels")
            
            if labels is None:
                raise RuntimeError("The processor did not produce labels. Cannot calculate supervised training loss.")

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
        
        if TRAIN_SUBSET_FRACTION < 1.0:
            print(f"Smoke test complete ({TRAIN_SUBSET_FRACTION*100}% of train set); skipping full validation.")
            return

        dev_loss, dev_wer = validate_qwen(
            model,
            processor,
            dev_loader,
            device,
            amp_enabled,
            amp_dtype,
        )
        
        save_checkpoint(model, processor, output_dir, global_epoch)

        train_losses.append(epoch_train_loss)
        dev_losses.append(dev_loss)
        dev_wers.append(dev_wer)

        print(f"\nEpoch {global_epoch} results:")
        print(f"  Training loss: {epoch_train_loss:.4f}")
        print(f"  Development loss: {dev_loss:.4f}")
        print(f"  Development WER: {dev_wer:.2%}")

        save_loss_plot(
            train_batch_losses,
            train_losses,
            dev_losses,
            output_dir,
            global_epoch,
        )

        metrics_path = (
            output_dir / f"training_metrics_{global_epoch}.csv"
        )

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
    
    torch.manual_seed(SEED)
    
    output_dir = MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output dir: {output_dir}")
    
    print("\nLoading Latest Checkpoint...")
    model, processor, device, starting_epoch = load_latest_checkpoint()
    
    print(f"Continuing from epoch: {starting_epoch}")
    print(f"Traininng for {NUM_EPOCHS} additional epochs.")
    
    print("\nLoading Manifests...")
    train_data = load_manifest(TRAIN_MANIFEST_CSV, "train")
    dev_data = load_manifest(DEV_MANIFEST_CSV, "dev")
    
    check_files(train_data, "train")
    check_files(dev_data, "dev")
    
    print("\nStarting Qwen fine-tuning and validation...")
    train_qwen(model, processor, train_data, dev_data, output_dir, device, starting_epoch)
    
    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed / 60:.2f} minutes.")
    print(f"Results saved under: {output_dir}")



# === Call Main ===

if __name__=="__main__":
    main()
