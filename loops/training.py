import torch
from callbacks.early_stopping import EarlyStopping
def training_loop(model, training_loader, validation_loader, optimizer, scheduler, loss_fn, experiment_config):
    """
    Training loop for the Whisper model.

    Args:
        model (torch.nn.Module): The Whisper model to train.
        training_loader (DataLoader): DataLoader for the training dataset.
        validation_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        loss_fn (callable): Loss function.
        device (str): Device to run the training on ('cpu' or 'cuda').
    """ 
    early_stopping = EarlyStopping(patience=experiment_config['early_stopping']['patience'], verbose=True, path=experiment_config['early_stopping']['model_savepath'])
    device = experiment_config['training']['device']    
    num_epochs = experiment_config['training']['num_train_epochs']  
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in training_loader:
            input_features = batch['input_features'].to(device)
            dec_input_ids = batch['dec_input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_features, dec_input_ids)

            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(training_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                input_features = batch['input_features'].to(device)
                dec_input_ids = batch['dec_input_ids'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_features, dec_input_ids)

                val_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(validation_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}") 

        early_stopping(val_loss, model, epoch)
    
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break    