import torch
import torch.nn as nn
import os 
import sys  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from whisper import load_model, tokenizer
import evaluate
from dataset.dataset import ASRDataset,WhisperDataCollatorWithPadding
import json
import whisper

class WhisperModel(nn.Module):
    def __init__(self, experiment_config, tokenizer):
        super().__init__()
        self.model_name = experiment_config["model"]['model_name']
        self.lang = experiment_config["model"]["lang"]
        self.device = experiment_config["training"]["device"]
        self.tokenizer = tokenizer
        self.model = load_model(self.model_name, device = self.device)
        for p in self.model.encoder.parameters():
            p.requires_grad = False

    def forward(self, input_ids, dec_input_ids=None):
        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)
        
        if dec_input_ids is not None:
            out = self.model.decoder(dec_input_ids, audio_features)
            return out
        return audio_features

    def predict(self, logits):
        logits = logits.clone()
        logits[logits == -100] = self.tokenizer.eot
        pred_tokens = torch.argmax(logits, dim=-1)
        decoded_preds = []
        special_token_ids = set(self.tokenizer.special_tokens.values())

        for t in pred_tokens:
            filtered = [token_id for token_id in t.cpu().tolist() if token_id not in special_token_ids]
            decoded_preds.append(self.tokenizer.decode(filtered))
        return decoded_preds


if __name__ == '__main__':

    with open("experiment_config.json", "r", encoding="utf-8") as f:
        experiment_config = json.load(f)

    dataset = ASRDataset(split='train', tokenizer=whisper.tokenizer.get_tokenizer(language=experiment_config["model"]["lang"],
                                                                                   task=experiment_config["model"]["task"],
                                                                                   multilingual=True))
    device = experiment_config["training"]["device"]    
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=WhisperDataCollatorWithPadding())
    model = WhisperModel(experiment_config, tokenizer)
    for batch in loader:
        input_features = batch['input_features'].to(device)
        dec_input_ids = batch['dec_input_ids'].to(device)
        outputs = model(input_features, dec_input_ids)
        print(outputs.shape)
        break


        

