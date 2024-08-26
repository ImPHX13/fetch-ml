import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SentenceTransformer(nn.Module):
    """
    A sentence transformer model based on BERT for generating sentence embeddings.
    """

    def __init__(self, bert_model_name='bert-base-uncased', embedding_dim=768):
        """
        Initialize the SentenceTransformer model.

        Args:
            bert_model_name (str): Name of the pre-trained BERT model to use.
            embedding_dim (int): Dimension of the output sentence embedding.
        """
        super(SentenceTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, embedding_dim)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tokenized input sentences.
            attention_mask (torch.Tensor): Attention mask for the input.

        Returns:
            torch.Tensor: Sentence embedding.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        sentence_embedding = self.fc(pooled_output)
        return sentence_embedding
    
    def encode(self, sentences):
        """
        Encode a list of sentences into embeddings.

        Args:
            sentences (List[str]): List of input sentences.

        Returns:
            torch.Tensor: Tensor of sentence embeddings.
        """
        # Tokenize and encode the input sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        # Generate embeddings without gradient computation
        with torch.no_grad():
            sentence_embeddings = self.forward(encoded_input['input_ids'], encoded_input['attention_mask'])
        
        return sentence_embeddings

# Test the implementation
if __name__ == "__main__":
    model = SentenceTransformer()
    sample_sentences = [
        "Sebastian Vettel is a 4 time world champion",
        "Michael Schumacher is the greatest driver of all time",
        "Max Verstappen is the current world champion"
    ]
    embeddings = model.encode(sample_sentences)
    print("Sample sentence embeddings:")
    print(embeddings)
    print("Embedding shape:", embeddings.shape)

"""
Key architectural choices:

1. Embedding dimension: Default 768 (BERT-base hidden size), adjustable via linear layer.
2. Pooling: Using BERT's [CLS] token representation (pooled output).
3. Additional layer: Single linear layer for dimensionality adjustment and task-specific learning.
4. Fine-tuning: BERT layers not frozen by default, allowing full model fine-tuning.
5. Tokenization: Using BERT tokenizer for consistency, integrated into the encode method.

These choices balance simplicity, flexibility, and effectiveness for a general-purpose
sentence transformer model.
"""