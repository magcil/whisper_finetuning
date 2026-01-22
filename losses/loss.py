import torch

def get_loss_fn(loss_name):
    """
    Get the cross-entropy loss function for Whisper model training.
    
    Returns:
        torch.nn.CrossEntropyLoss: Loss function with ignore_index set to -100.
    """
    if loss_name == "CE":
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    else:
        pass  # TODO: implement later with other loss functions
    
    return loss_fn
    