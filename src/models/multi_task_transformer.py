import torch
import torch.nn as nn
from .sentence_transformer import SentenceTransformer

class MultiTaskTransformer(nn.Module):
    def __init__(self, num_classes_a, num_classes_b, bert_model_name='bert-base-uncased', embedding_dim=768):
        super(MultiTaskTransformer, self).__init__()
        self.sentence_transformer = SentenceTransformer(bert_model_name, embedding_dim)
        self.classifier_a = nn.Linear(embedding_dim, num_classes_a)
        self.classifier_b = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_classes_b)
        )
        
    def forward(self, input_ids, attention_mask):
        sentence_embedding = self.sentence_transformer(input_ids, attention_mask)
        output_a = self.classifier_a(sentence_embedding)
        output_b = self.classifier_b(sentence_embedding)
        return output_a, output_b

# Test the implementation
if __name__ == "__main__":
    num_classes_a = 3  # Sentence classification classes
    num_classes_b = 5  # Named Entity Recognition classes (for example)
    
    model = MultiTaskTransformer(num_classes_a, num_classes_b)
    
    # Sample input
    input_ids = torch.randint(0, 1000, (2, 10))  # Batch size 2, sequence length 10
    attention_mask = torch.ones_like(input_ids)
    
    output_a, output_b = model(input_ids, attention_mask)
    print("Task A output shape:", output_a.shape)
    print("Task B output shape:", output_b.shape)

"""
Changes made to the architecture to support multi-task learning:

1. Shared Backbone: We reused the SentenceTransformer as the shared backbone for both tasks.
   This allows the model to learn common features that are useful for both tasks, promoting
   efficient use of parameters and potential positive transfer between tasks.

2. Task-Specific Heads: We added two separate classifier heads (classifier_a and classifier_b)
   on top of the shared sentence embedding. Each classifier is a linear layer that maps the
   sentence embedding to the respective task's output space:
   - classifier_a: For sentence classification (Task A)
   - classifier_b: For the second NLP task, e.g., Named Entity Recognition (Task B)

3. Multi-Output Forward Pass: The forward method now returns outputs for both tasks
   simultaneously. This allows for joint training on both tasks in a single forward pass.

4. Flexible Class Numbers: The constructor takes num_classes_a and num_classes_b as
   parameters, allowing flexibility in the number of classes for each task.

5. Shared Embedding Space: Both tasks operate on the same sentence embedding, encouraging
   the model to learn a unified representation that's useful for both tasks.

6. No Task-Specific Feature Engineering: By using the same sentence embedding for both
   tasks, we rely on the model to learn task-specific features implicitly, rather than
   engineering separate features for each task.

These changes enable the model to perform multi-task learning while maintaining a
relatively simple architecture. The shared backbone promotes parameter efficiency
and potential knowledge transfer between tasks, while the task-specific heads allow
for specialization to each task's requirements. This architecture can be easily
extended to more than two tasks by adding additional classifier heads as needed.
"""