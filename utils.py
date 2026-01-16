import torchaudio
import torchaudio.transforms as at
import torch
from torch.optim import AdamW

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

def get_optimizer_and_scheduler(model,experiment_config, train_dataset):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in self.model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": experiment_config['training']['weight_decay'],
        },
        {
            "params": [p for n, p in self.model.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }]

    self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=experiment_config['training']['learning_rate'],
            eps=experiment_config['training']['adam_epsilon'])

    training_steps = len(train_dataset)// experiment_config['training']['batch_size'] // experiment_config['training']['gradient_accumulation_steps']* float(experiment_config['training']['num_train_epochs'])
                
    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=experiment_config['training']['warmup_steps'],
            num_training_steps=training_steps)
    
    return optimizer, scheduler
