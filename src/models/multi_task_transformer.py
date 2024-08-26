import torch
import torch.nn as nn
from .sentence_transformer import SentenceTransformer

class MultiTaskTransformer(nn.Module):
    """
    A multi-task transformer model that performs two classification tasks simultaneously.
    
    This model uses a shared SentenceTransformer as a backbone and has separate classifier
    heads for each task. It's designed for combined learning of related NLP tasks.
    """

    def __init__(self, num_classes_a, num_classes_b, bert_model_name='bert-base-uncased', embedding_dim=768):
        """
        Initialize the MultiTaskTransformer.

        Args:
            num_classes_a (int): Number of classes for task A (sentence classification)
            num_classes_b (int): Number of classes for task B (named entity recognition)
            bert_model_name (str): Name of the pre-trained BERT model to use
            embedding_dim (int): Dimension of the sentence embedding
        """
        super(MultiTaskTransformer, self).__init__()
        
        # Shared backbone for feature extraction
        self.sentence_transformer = SentenceTransformer(bert_model_name, embedding_dim)
        
        # Task A: Simple linear classifier
        self.classifier_a = nn.Linear(embedding_dim, num_classes_a)
        
        # Task B: Multi-layer classifier with ReLU activation
        self.classifier_b = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_classes_b)
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Token IDs of input sequences
            attention_mask (torch.Tensor): Attention mask for input sequences

        Returns:
            tuple: (output_a, output_b) containing logits for both tasks
        """
        # Get sentence embedding from shared backbone
        sentence_embedding = self.sentence_transformer(input_ids, attention_mask)
        
        # Task A: Apply linear classifier
        output_a = self.classifier_a(sentence_embedding)
        
        # Task B: Apply multi-layer classifier
        output_b = self.classifier_b(sentence_embedding)
        
        return output_a, output_b

# Test the implementation
if __name__ == "__main__":
    num_classes_a = 3  # Sentence classification classes
    num_classes_b = 5  # Named Entity Recognition classes
    
    # Initialize the multi-task model
    model = MultiTaskTransformer(num_classes_a, num_classes_b)
    
    # Create sample input
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones_like(input_ids)
    
    # Perform a forward pass
    output_a, output_b = model(input_ids, attention_mask)
    
    # Print output shapes
    print("Sentence classification output shape:", output_a.shape)
    print("Named Entity Recognition output shape:", output_b.shape)

"""
Key changes for multi-task learning:

1. Shared Backbone: SentenceTransformer serves as a common feature extractor for both tasks, 
   promoting efficient parameter use and potential knowledge transfer.

2. Task-Specific Classifiers: Separate classifier heads (classifier_a and classifier_b) 
   allow the model to specialize for each task while sharing common features.

3. Multi-Output Forward Pass: The forward method returns outputs for both tasks simultaneously, 
   enabling joint training in a single pass.

4. Flexible Class Configuration: Constructor parameters num_classes_a and num_classes_b 
   allow easy adaptation to different task requirements.

5. Unified Embedding: Both tasks operate on the same sentence embedding, encouraging 
   the model to learn a versatile representation useful for multiple tasks.

This architecture balances shared learning and task-specific specialization, 
and can be easily extended to more tasks by adding classifier heads as needed.
"""