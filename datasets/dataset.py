import torch
from torch.utils.data import Dataset
import os 
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from utils import load_wave
import whisper
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Any

class ASRDataset(Dataset):
    """
    Dataset class for Automatic Speech Recognition (ASR) tasks.
    
    Each item consists of:
      - Mel spectrogram of an audio file
      - Tokenized transcription (labels)
      - Decoder input sequence for training
    
    Attributes:
        data_list (list): List of dicts with keys 'audio_path' and 'text'
        sample_rate (int): Sampling rate to load audio
        tokenizer: Tokenizer for transcriptions
    """
    
    def __init__(self, split: str, tokenizer, sample_rate: int = 16000) -> None:
        """
        Args:
            split (str): Name of the dataset split (expects 'split.json')
            tokenizer: Tokenizer object compatible with Whisper
            sample_rate (int): Sampling rate to load audio (default 16kHz)
        """
        super().__init__()
        
        # Load the JSON split file containing audio paths and transcriptions
        split_path = f'data/{split}.json'
        try:
            with open(split_path, "r", encoding="utf-8") as f:
                self.data_list = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Split file not found: {split_path}")
        
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single training example.
        
        Steps:
        1. Load the waveform from disk
        2. Ensure the audio is not stereo
        3. Trim/pad audio to 30 seconds for Whisper
        4. Convert to log-mel spectrogram
        5. Tokenize the transcription for decoder input and labels
        
        Args:
            idx (int): Index of the example
            
        Returns:
            dict: {
                "input_ids": Mel spectrogram tensor,
                "dec_input_ids": Decoder input IDs,
                "labels": Target labels for loss computation
            }
        """
        # Extract file path and text transcription
        audio_path, text_path = self.data_list[idx].values()
        with open(text_path, mode='r') as text_file:
            text = text_file.read()
        
        # Load audio waveform
        audio = load_wave(wave_path=audio_path, sample_rate=self.sample_rate)

        # Flatten and pad/trim audio to 30s for Whisper
        audio = whisper.pad_or_trim(audio.flatten())
        
        # Convert audio to log-mel spectrogram
        mel_spec = whisper.log_mel_spectrogram(audio)
        
        # Tokenize transcription
        text_ids = self.tokenizer.encode(text)
        
        # Decoder input sequence during training
        dec_input_ids = [*self.tokenizer.sot_sequence_including_notimestamps] + text_ids
        
        # Labels (ground truth) for loss computation
        # skip bos token
        labels = dec_input_ids[1:] + [self.tokenizer.eot]
        
        return {
            "input_features": mel_spec,
            "dec_input_ids": dec_input_ids,
            "labels": labels
        }

class WhisperDataCollatorWithPadding:
    """
    Data collator for Whisper-style sequence-to-sequence training.
    
    This collator:
      - Stacks precomputed log-mel spectrograms
      - Pads decoder input IDs and labels to the same length
      - Uses -100 for label padding so padded tokens are ignored by the loss
      - Uses the EOT token (50257) for decoder input padding
    
    Expected input format (per example):
        {
            "input_features": torch.Tensor,   # shape: [n_mels, time]
            "dec_input_ids": List[int],
            "labels": List[int]
        }
    """
    
    def __call__(self, input_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of Whisper training examples.
        
        Args:
            input_data (List[Dict]): List of dataset items.
            
        Returns:
            Dict[str, torch.Tensor]: Batched and padded tensors:
                - input_features: Tensor of shape [B, n_mels, time]
                - dec_input_ids: Tensor of shape [B, max_seq_len]
                - labels: Tensor of shape [B, max_seq_len]
        """
        # Extract audio features (assumed already padded to same length)
        input_features = [d["input_features"] for d in input_data]
        
        # Convert token lists to tensors
        labels = [torch.tensor(d["labels"], dtype=torch.long) for d in input_data]
        dec_input_ids = [
            torch.tensor(d["dec_input_ids"], dtype=torch.long) for d in input_data
        ]

        # Stack audio features into a batch tensor
        input_features = torch.stack(input_features, dim=0)

        # Compute the maximum sequence length across labels and decoder inputs
        max_len = max(
            max(l.size(0) for l in labels),
            max(d.size(0) for d in dec_input_ids),
        )

        # Pad labels with -100 so they are ignored by the loss function
        labels = torch.stack([
            F.pad(l, (0, max_len - l.size(0)), value=-100)
            for l in labels
        ])

        # Pad decoder input IDs with the Whisper EOT token (50257)
        dec_input_ids = torch.stack([
            F.pad(d, (0, max_len - d.size(0)), value=50257)
            for d in dec_input_ids
        ])

        return {
            "input_features": input_features,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
        }
if __name__ == '__main__':
    dataset = ASRDataset(split='train', tokenizer=whisper.tokenizer.get_tokenizer(language='el',
                                                                                   task='transcribe',
                                                                                   multilingual=True))
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=3, collate_fn=WhisperDataCollatorWithPadding())
    for i in loader:
        print(i['dec_input_ids'])
        print(i['labels'])
        break


