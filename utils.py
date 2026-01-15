import torchaudio
import torchaudio.transforms as at
import torch

def load_wave(wave_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """
    Load an audio file as a waveform tensor.
    
    Args:
        wave_path (str): Path to the audio file.
        sample_rate (int): Desired sample rate for the waveform (default 16kHz).
        
    Returns:
        torch.Tensor: Audio waveform tensor with shape [channels, time].
        
    Raises:
        FileNotFoundError: If the audio file does not exist.
        AssertionError: If the audio is stereo (2 channels).
    """
    import os
    
    # Check that file exists
    if not os.path.exists(wave_path):
        raise FileNotFoundError(f"Audio file not found: {wave_path}")
    
    # Load waveform
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    
    # Ensure the audio is not stereo (2 channels)
    assert waveform.shape[0] != 2, f"Audio has 2 channels (stereo), must be mono. Shape: {waveform.shape}"
    
    # Resample if needed
    if sr != sample_rate:
        waveform = at.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    
    return waveform