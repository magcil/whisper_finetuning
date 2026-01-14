import numpy as np

import torch
from torch import nn
import pandas as pd
import whisper
import torchaudio
import torchaudio.transforms as at
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.notebook import tqdm
import evaluate
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


if "name" == "__main__":
    print("ELA P EIS")