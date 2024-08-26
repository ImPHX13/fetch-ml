import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class LayerwiseLearningRate(_LRScheduler):
    """Custom learning rate scheduler for layer-wise learning rates."""

    def __init__(self, optimizer, layer_lrs, last_epoch=-1):
        self.layer_lrs = layer_lrs
        super(LayerwiseLearningRate, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [lr for lr in self.layer_lrs]

def set_layerwise_lr(model, lr_backbone, lr_heads):
    """
    Set different learning rates for different parts of the model.

    Args:
        model: The multi-task transformer model
        lr_backbone: Learning rate for the BERT backbone
        lr_heads: Learning rate for the task-specific heads

    Returns:
        optimizer: AdamW optimizer with grouped parameters
        scheduler: LayerwiseLearningRate scheduler
    """
    optimizer_grouped_parameters = [
        {'params': model.sentence_transformer.bert.parameters(), 'lr': lr_backbone},
        {'params': model.sentence_transformer.fc.parameters(), 'lr': lr_heads},
        {'params': model.classifier_a.parameters(), 'lr': lr_heads},
        {'params': model.classifier_b.parameters(), 'lr': lr_heads},
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    scheduler = LayerwiseLearningRate(optimizer, [lr_backbone, lr_heads, lr_heads, lr_heads])
    
    return optimizer, scheduler

# Example usage
if __name__ == "__main__":
    from ..models.multi_task_transformer import MultiTaskTransformer
    
    model = MultiTaskTransformer(num_classes_a=3, num_classes_b=5)
    lr_backbone = 1e-5
    lr_heads = 1e-4
    
    optimizer, scheduler = set_layerwise_lr(model, lr_backbone, lr_heads)
    
    print("Learning rates:")
    for param_group in optimizer.param_groups:
        print(f"{param_group['lr']:.1e}")

"""
Rationale for the specific learning rates:

1. BERT backbone (lr_backbone = 1e-5):
   - Lower learning rate to preserve pre-trained knowledge
   - Allows gentle fine-tuning without losing valuable information

2. Task-specific layers (lr_heads = 1e-4):
   - Higher learning rate for faster adaptation to new tasks
   - These layers are newly initialized and need to learn quickly

Benefits of layer-wise learning rates in multi-task setting:

1. Balances stability and adaptability
2. Faster convergence for task-specific layers
3. Preserves shared knowledge in the backbone
4. Allows more control over optimization
5. Facilitates balanced learning across multiple tasks

This approach helps optimize multi-task transformer models by tailoring 
the learning process to different parts of the network and task requirements.
"""