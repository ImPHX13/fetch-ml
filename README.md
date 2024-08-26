# Multi-Task Sentence Transformer

This project implements a multi-task sentence transformer model with layer-wise learning rates.

## Running with Docker

To run this project using Docker:

1. Build the Docker image:
   ```
   docker build -t multi-task-transformer .
   ```

2. Run the Docker container:
   ```
   docker run multi-task-transformer
   ```

This will execute the `main.py` script, which demonstrates the Sentence Transformer, Multi-Task Transformer, and Layer-wise Learning Rate implementation.

## Project Structure

- `src/models/sentence_transformer.py`: Implementation of the Sentence Transformer
- `src/models/multi_task_transformer.py`: Implementation of the Multi-Task Transformer
- `src/train/layer_wise_lr.py`: Implementation of Layer-wise Learning Rate
- `main.py`: Main script that orchestrates and demonstrates all components

## Task 3: Training Considerations - Key Decisions and Insights

When considering different freezing scenarios for our multi-task transformer model, we made the following key decisions:

1. Freezing the entire network:
   - Best for scenarios with very limited task-specific data
   - Preserves pre-trained knowledge but limits adaptation to new tasks
   - Useful as a baseline or for pure feature extraction

2. Freezing only the transformer backbone:
   - Balances preservation of general language understanding with task-specific adaptation
   - Allows task-specific heads to learn while maintaining the core language model
   - Recommended for moderate amounts of task-specific data

3. Freezing one task-specific head:
   - Enables fine-tuning for a new task while maintaining performance on an existing task
   - Useful for incremental learning or when one task is well-optimized
   - Requires careful monitoring to prevent imbalanced performance across tasks

For transfer learning, we chose to:
- Use a large pre-trained model like BERT or RoBERTa as our starting point
- Freeze lower layers and unfreeze upper layers of the transformer
- Fine-tune task-specific heads from scratch

This approach leverages general language understanding while allowing adaptation to specific tasks, balancing efficiency and performance.

## Task 4: Layer-wise Learning Rate - Key Decisions and Insights

Implementing layer-wise learning rates for our multi-task transformer led to several important insights:

1. Learning rate differentiation:
   - Lower learning rate (e.g., 1e-5) for pre-trained layers to preserve valuable knowledge
   - Higher learning rate (e.g., 1e-4) for task-specific layers to enable faster adaptation

2. Benefits in multi-task setting:
   - Shared backbone adapts slowly, benefiting all tasks
   - Task-specific heads adapt quickly to individual task requirements
   - Balances knowledge sharing and task-specific optimization

3. Flexibility and control:
   - Allows fine-grained optimization strategies
   - Can prioritize certain tasks or layers if needed
   - Helps mitigate issues like vanishing gradients in deep networks

4. Transfer learning facilitation:
   - Enables controlled knowledge transfer between tasks
   - Allows for easy adjustment of learning process based on task similarities or differences

By implementing layer-wise learning rates, we gained a powerful tool for optimizing our multi-task model, allowing us to balance the needs of different layers and tasks within a single architecture. This approach provides the flexibility to adapt our training strategy to various scenarios and dataset characteristics, potentially leading to improved performance and more efficient training.