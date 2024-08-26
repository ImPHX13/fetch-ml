import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class LayerwiseLearningRate(_LRScheduler):
    def __init__(self, optimizer, layer_lrs, last_epoch=-1):
        self.layer_lrs = layer_lrs
        super(LayerwiseLearningRate, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [lr for lr in self.layer_lrs]

def set_layerwise_lr(model, lr_backbone, lr_heads):
    # Set different learning rates for different parts of the model
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

1. Backbone (BERT) layers (lr_backbone = 1e-5):
   - We use a lower learning rate for the pre-trained BERT layers.
   - Rationale: These layers have already learned general language representations during 
     pre-training. We want to fine-tune them gently to avoid catastrophic forgetting of 
     this valuable information while allowing some adaptation to our specific tasks.

2. Task-specific layers (lr_heads = 1e-4):
   - We use a higher learning rate for the FC layer and task-specific classifiers.
   - Rationale: These layers are newly initialized for our specific tasks. They need to 
     learn more quickly to adapt to the task requirements, as they don't have the benefit 
     of pre-training.

Potential benefits of using layer-wise learning rates:

1. Preserves pre-trained knowledge: Lower learning rates for pre-trained layers help maintain 
   the valuable information learned during pre-training.
2. Faster adaptation to new tasks: Higher learning rates for task-specific layers allow them 
   to adapt more quickly to the new tasks.
3. Balances stability and plasticity: This approach helps maintain a balance between retaining 
   useful pre-trained features and learning task-specific features.
4. Mitigates vanishing gradients: By allowing higher learning rates in upper layers, we can 
   ensure that these layers receive meaningful updates even if gradients are small.
5. Fine-grained control: Allows for more precise optimization strategies, potentially leading 
   to better convergence and performance.

How the multi-task setting plays into layer-wise learning rates:

1. Shared knowledge utilization: In a multi-task setting, the lower learning rate for the 
   shared backbone (BERT) allows all tasks to benefit from the pre-trained knowledge while 
   slowly adapting it to the common features of all tasks.
2. Task-specific adaptation: The higher learning rate for task-specific heads allows each 
   task to quickly adapt to its specific requirements without interfering with the other task.
3. Balancing task influence: By using the same learning rate for all task-specific heads, 
   we ensure that no single task dominates the learning process, promoting a balanced 
   multi-task learning scenario.
4. Transfer learning facilitation: The layer-wise approach can help in scenarios where one 
   task might benefit from transferring knowledge from another task, by allowing fine-grained 
   control over which parts of the network adapt more quickly to each task.
5. Flexibility in task importance: If needed, we could easily adjust the learning rates of 
   individual task heads to prioritize certain tasks over others, which is particularly 
   useful in multi-task scenarios where tasks may have different levels of importance or 
   difficulty.

This layer-wise learning rate approach provides a flexible and powerful tool for optimizing 
multi-task transformer models, allowing us to balance the needs of different layers and 
tasks within a single model.
"""