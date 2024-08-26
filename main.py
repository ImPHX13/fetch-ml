import torch
from src.models.sentence_transformer import SentenceTransformer
from src.models.multi_task_transformer import MultiTaskTransformer
from src.train.layer_wise_lr import set_layerwise_lr

def main():
    print("Demonstrating Sentence Transformer:")
    sentence_transformer = SentenceTransformer()
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "I love natural language processing!",
        "Transformers have revolutionized NLP tasks."
    ]
    embeddings = sentence_transformer.encode(sample_sentences)
    print("Sample sentence embeddings:")
    print(embeddings)
    print("Embedding shape:", embeddings.shape)

    print("\nDemonstrating Multi-Task Transformer:")
    num_classes_a = 3  # Sentence classification classes
    num_classes_b = 5  # Named Entity Recognition classes (for example)
    multi_task_model = MultiTaskTransformer(num_classes_a, num_classes_b)
    
    # Sample input
    input_ids = torch.randint(0, 1000, (2, 10))  # Batch size 2, sequence length 10
    attention_mask = torch.ones_like(input_ids)
    
    output_a, output_b = multi_task_model(input_ids, attention_mask)
    print("Task A output shape:", output_a.shape)
    print("Task B output shape:", output_b.shape)

    print("\nDemonstrating Layer-wise Learning Rate:")
    lr_backbone = 1e-5
    lr_heads = 1e-4
    
    optimizer, scheduler = set_layerwise_lr(multi_task_model, lr_backbone, lr_heads)
    
    print("Learning rates:")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Layer {i}: {param_group['lr']:.1e}")

if __name__ == "__main__":
    main()