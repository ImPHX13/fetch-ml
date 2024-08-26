import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SentenceTransformer(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', embedding_dim=768):
        super(SentenceTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, embedding_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        sentence_embedding = self.fc(pooled_output)
        return sentence_embedding
    
    def encode(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            sentence_embeddings = self.forward(encoded_input['input_ids'], encoded_input['attention_mask'])
        return sentence_embeddings

# Test the implementation
if __name__ == "__main__":
    model = SentenceTransformer()
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "I love natural language processing!",
        "Transformers have revolutionized NLP tasks."
    ]
    embeddings = model.encode(sample_sentences)
    print("Sample sentence embeddings:")
    print(embeddings)
    print("Embedding shape:", embeddings.shape)

"""
Choices made regarding the model architecture outside of the transformer backbone:

1. Embedding dimension: We chose to use a default embedding dimension of 768, which matches
   the hidden size of BERT-base. This allows us to directly use the output of BERT's pooler
   layer if needed. However, we added a linear layer (self.fc) to allow flexibility in
   changing the embedding dimension if required.

2. Pooling strategy: We used the pooled output from BERT, which is typically the [CLS] token
   representation. This choice was made because it's a common and effective way to get a
   fixed-length representation of the entire sentence. Alternative strategies could include
   mean pooling or max pooling over all token embeddings.

3. Additional layers: We chose to use a single linear layer (self.fc) after the BERT output.
   This allows for dimensionality reduction if needed and adds a small amount of task-specific
   learning capacity. We could have added more complexity here, such as multiple layers or
   non-linearities, but opted for simplicity given that BERT already provides rich representations.

4. No fine-tuning flag: We didn't include an option to freeze BERT layers. This could be added
   as a parameter to the constructor if we want to use the model purely as a feature extractor
   without fine-tuning.

5. Tokenization: We use the BERT tokenizer for consistency with the pre-trained model. The
   tokenization is handled in the encode method, which allows for easy use of the model with
   raw text input.

These choices balance simplicity, flexibility, and effectiveness for a general-purpose
sentence transformer model. Depending on the specific use case, these could be adjusted
or expanded upon.
"""