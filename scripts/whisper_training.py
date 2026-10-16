import whisper
import torch
import whisper.audio as whisper_audio
from pathlib import Path
import pandas as pd
import re


SIZE = "medium" # medium or turbo
N_MELS = 80 # 80 or 128
INPUT_CSV = Path("data/manifests/m3av_train_manifest.csv")
DATA_DIR = Path("data/processed/segments")


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = whisper.load_model(SIZE, device=device)
    return model


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


def extract_audio_features(audio_file):
    # find the number of samples for the audio file
    n_samples = whisper_audio.load_audio(audio_file).shape[0]
    padding_length = whisper_audio.N_SAMPLES - n_samples
    return whisper_audio.log_mel_spectrogram(audio_file, n_mels=N_MELS, padding=padding_length)


# check validity of audiofile
def is_valid_wav(audio_file):
    try:
        whisper_audio.load_audio(audio_file)
        return True
    except Exception as e:
        print(f"[skip] {audio_file}: \n{e}")
        return False


# define data loader
class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, audio_files, transcriptions):
        valid = [(a, t) for a, t in zip(audio_files, transcriptions) if is_valid_wav(a)]
        self.audio_files, self.transcriptions = zip(*valid) if valid else ([], [])
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        mel = extract_audio_features(self.audio_files[idx])
        return mel, self.transcriptions[idx]


def collate_fn(batch):
    mels, transcriptions = zip(*batch)
    return torch.stack(mels), transcriptions


def get_files_and_transcriptions(input_csv):
    # TODO: change paths in two places
    df = pd.read_csv(input_csv)
    required_columns = {"sample_id", "reference"}
    
    missing = required_columns - set(df.columns)
    if missing: 
        raise ValueError(f"There are missing columns in manifest.\nMissing Columns: {missing}")
    if df["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample IDs.")
    
    audio_files = [f"{DATA_DIR}/train/{i}.wav" for i in df["sample_id"].tolist()]
    transcriptions = df["reference"].apply(normalize_text).tolist()
    
    return audio_files, transcriptions


# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class LabelSmoothingLoss(torch.nn.Module):
    """
    Implement the LabelSmoothingLoss proposed in the following paper
    https://arxiv.org/pdf/1512.00567.pdf
    (Rethinking the Inception Architecture for Computer Vision)
    """
    def __init__(
        self,
        ignore_index: int = -1,
        label_smoothing: float = 0.1,
        reduction: str = "sum",
    ) -> None:
        """
        Args:
          ignore_index:
            ignored class id
          label_smoothing:
            smoothing rate (0.0 means the conventional cross entropy loss)
          reduction:
            It has the same meaning as the reduction in
            `torch.nn.CrossEntropyLoss`. It can be one of the following three
            values: (1) "none": No reduction will be applied. (2) "mean": the
            mean of the output is taken. (3) "sum": the output will be summed.
        """
        super().__init__()
        assert 0.0 <= label_smoothing < 1.0, f"{label_smoothing}"
        assert reduction in ("none", "sum", "mean"), reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between x and target.
        Args:
          x:
            prediction of dimension
            (batch_size, input_length, number_of_classes).
          target:
            target masked with self.ignore_index of
            dimension (batch_size, input_length).
        Returns:
          A scalar tensor containing the loss without normalization.
        """
        assert x.ndim == 3
        assert target.ndim == 2
        assert x.shape[:2] == target.shape
        num_classes = x.size(-1)
        x = x.reshape(-1, num_classes)
        # Now x is of shape (N*T, C)
        # We don't want to change target in-place below,
        # so we make a copy of it here
        target = target.clone().reshape(-1)
        ignored = target == self.ignore_index
        # See https://github.com/k2-fsa/icefall/issues/240
        # and https://github.com/k2-fsa/icefall/issues/297
        # for why we don't use target[ignored] = 0 here
        target = torch.where(ignored, torch.zeros_like(target), target)
        true_dist = torch.nn.functional.one_hot(target, num_classes=num_classes).to(x)
        true_dist = (
            true_dist * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        )
        # Set the value of ignored indexes to 0
        #
        # See https://github.com/k2-fsa/icefall/issues/240
        # and https://github.com/k2-fsa/icefall/issues/297
        # for why we don't use true_dist[ignored] = 0 here
        true_dist = torch.where(
            ignored.unsqueeze(1).repeat(1, true_dist.shape[1]),
            torch.zeros_like(true_dist),
            true_dist,
        )
        loss = -1 * (torch.log_softmax(x, dim=1) * true_dist)
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.sum() / (~ignored).sum()
        else:
            return loss.sum(dim=-1)


# DO NOT MODIFY
# Copyright    (authors: Xiaoyu Yang)
#               2023  Xiaomi Corp.        
#               2024  Yuekai Zhang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Any
from torch import Tensor
from torch.nn.functional import pad as pad_tensor
def batch_tensors(tensors: List[Tensor], pad_value: Any) -> Tensor:
    padding_size = max(tensor.shape[0] for tensor in tensors)
    dims = len(tensors[0].shape)
    padded_tensors = []
    for tensor in tensors:
        padding = [0] * 2 * dims
        padding[-1] = padding_size - tensor.shape[0]
        padded_tensors.append(pad_tensor(tensor, padding, "constant", pad_value))
    return torch.stack([tensor for tensor in padded_tensors], dim=0)


def train_whisper(model, train_dataset):

    # Find the start of sentence token in the vocabulary
    tokenizer = whisper.tokenizer.get_tokenizer(True, num_languages=model.num_languages, language="en", task="transcribe")
    tokens_to_prepend = tokenizer.encode(
    "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    allowed_special={"<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"})

    # Fine-tuning using GPU
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #TODO: move the model to the device
    model = model.to(device)

    # Optimizer
    #TODO: use Adam optimizer with learning rate 2e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Training loop
    for batch_idx, (feats, texts) in enumerate(train_loader):

    # Stop after running on 25% of the data
        if batch_idx >= len(train_loader) * 0.05:
            break
        
        # Normalize and convert text to tensor of indices using the tokenizer
        text_tokens_list = [
            tokens_to_prepend
            + tokenizer.encode(text.lower())
            + [tokenizer.eot]
            for text in texts
        ]
        
        # Convert to torch tensors
        text_tokens_list = [
            torch.LongTensor(text_tokens) for text_tokens in text_tokens_list
        ]
        
        # Pad sequences
        prev_outputs_tokens = batch_tensors(
            [tokens[:-1] for tokens in text_tokens_list], pad_value=50256
        ).to(device)
        target_tokens = batch_tensors(
            [tokens[1:] for tokens in text_tokens_list], pad_value=50256
        ).to(device)
        target_lengths = torch.LongTensor(
            [tokens.shape[0] - 1 for tokens in text_tokens_list]
        ).to(device)
        
        # Move feats to the same device as the model
        #TODO:
        feats = feats.to(device)
        
        # Zero gradients
        model.zero_grad()
        
        # Loss function
        ignore_prefix_size = 3
        decoder_criterion = LabelSmoothingLoss(
            ignore_index=50256, label_smoothing=0.1, reduction="sum"
        )
        
        # Forward pass
        encoder_out = model.encoder(feats)
        text_logits = model.decoder(prev_outputs_tokens, encoder_out)
        text_logits = text_logits[:, ignore_prefix_size:, :]
        target_tokens = target_tokens[:, ignore_prefix_size:]
        
        # Compute loss
        loss = decoder_criterion(text_logits, target_tokens)
        loss_value = loss.item()
        
        # Normalize loss
        loss_value_normalized = loss_value / target_tokens.shape[0]
        print(f"batch_idx: {batch_idx}, loss: {loss_value_normalized}")
        
        # Backpropagation
        #TODO:
        loss = loss/8
        loss.backward()
        
        # Optimizer step every 8 batches
        #TODO:
        if (batch_idx+1)%8==0:
            optimizer.step()
            optimizer.zero_grad()

    # Final optimizer step for any leftover gradients
    #TODO:
    optimizer.step()
    optimizer.zero_grad()
    
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        f"models/whisper_{SIZE}_finetuned.pt"
    )


def main():
    model = load_model()
    print("audioset start")
    audio_files, transcriptions = get_files_and_transcriptions(INPUT_CSV)
    train_dataset = AudioDataset(audio_files, transcriptions)
    train_whisper(model, train_dataset)
    


if __name__=="__main__":
    main()

