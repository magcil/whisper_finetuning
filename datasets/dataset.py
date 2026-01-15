import torch
from torch.utils.data import Dataset
import json
from utils import load_wave
import whisper

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
        split_path = f'splits/{split}.json'
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
        audio_path, text = self.data_list[idx].values()
        
        # Load audio waveform
        audio = load_wave(wave_path=audio_path, sample_rate=self.sample_rate)
        
        # Ensure audio is not stereo
        assert audio.shape[0] != 2, f"Audio has 2 channels (stereo), must be mono. Shape: {audio.shape}"
        
        # Flatten and pad/trim audio to 30s for Whisper
        audio = whisper.pad_or_trim(audio.flatten())
        
        # Convert audio to log-mel spectrogram
        mel_spec = whisper.log_mel_spectrogram(audio)
        
        # Tokenize transcription
        text_ids = self.tokenizer.encode(text)
        
        # Decoder input sequence during training
        dec_input_ids = [*self.tokenizer.sot_sequence_including_notimestamps] + text_ids
        
        # Labels (ground truth) for loss computation
        labels = dec_input_ids[1:] + [self.tokenizer.eot]
        
        return {
            "input_ids": mel_spec,
            "dec_input_ids": dec_input_ids,
            "labels": labels
        }
