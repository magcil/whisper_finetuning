import torch
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import WhisperModel
from dataset.dataset import ASRDataset, WhisperDataCollatorWithPadding
from loops.training import training_loop
from loops.testing import testing_loop
from utils import get_optimizer_and_scheduler
from torch.utils.data import DataLoader
from losses.loss import get_loss_fn
from whisper import tokenizer

## Load experiment configuration
with open("experiment_config.json", "r", encoding="utf-8") as f:
    experiment_config = json.load(f)    

## Configure tokenizer
tokenizer = tokenizer.get_tokenizer(True, language=experiment_config['model']['lang'], task=experiment_config['model']['task'])
## Initialize model     
model = WhisperModel(experiment_config, tokenizer)
## Prepare datasets and dataloaders
train_dataset = ASRDataset(split='train', tokenizer=tokenizer)
val_dataset = ASRDataset(split='validation', tokenizer=tokenizer)
test_dataset = ASRDataset(split='test', tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=experiment_config['training']['batch_size'], collate_fn=WhisperDataCollatorWithPadding(), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=experiment_config['training']['batch_size'], collate_fn=WhisperDataCollatorWithPadding())
test_loader = DataLoader(test_dataset, batch_size=experiment_config['training']['batch_size'], collate_fn=WhisperDataCollatorWithPadding())

## Setup optimizer and scheduler
optimizer, scheduler = get_optimizer_and_scheduler(model, experiment_config, train_dataset) 
## Define loss function
loss_fn = get_loss_fn(experiment_config['training']['loss_fn'])
## Start training loop
training_loop(model, train_loader, val_loader, optimizer, scheduler, loss_fn, experiment_config)
## Load the best model after training
trained_model = WhisperModel(experiment_config, tokenizer)
trained_model.load_state_dict(torch.load(experiment_config['early_stopping']['model_savepath']))
## Evaluate on test set
test_results = testing_loop(trained_model, test_loader, experiment_config['training']['device'], compute_metrics=True)
print("Test Results:", test_results)    
