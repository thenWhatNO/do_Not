import numpy as np


#/////////// Multi-Head Attention ////////////////////

def softmax(logits):
    logits_exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)

# Split the matrices for multi-heads
def split_heads(X, num_heads):
    """
    Split the last dimension into (num_heads, depth_per_head).
    """
    batch_size, seq_length, d_model = X.shape
    depth_per_head = d_model // num_heads
    X = X.reshape(batch_size, seq_length, num_heads, depth_per_head)

    return np.transpose(X, axes=(0, 2, 1, 3))

# Combine heads after attention
def combine_heads(X):
    """
    Combine multi-head outputs into a single matrix.
    """
    batch_size, num_heads, seq_length, depth_per_head = X.shape
    d_model = num_heads * depth_per_head
    X = np.transpose(X, axes=(0, 2, 1, 3))
    return X.reshape(batch_size, seq_length, d_model)

# Inputs
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])[None, :, :]  # Shape (1, 2, 4)
K = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])[None, :, :]  # Shape (1, 2, 4)
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])[None, :, :]  # Shape (1, 2, 4)
W = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])  # Final projection

num_heads = 2  # Number of attention heads
d_model = Q.shape[-1]
depth_per_head = d_model // num_heads

# Step 1: Split into heads
Q_heads = split_heads(Q, num_heads)  # Shape (1, 2, 2, 2)
K_heads = split_heads(K, num_heads)  # Shape (1, 2, 2, 2)
V_heads = split_heads(V, num_heads)  # Shape (1, 2, 2, 2)

# Step 2: Scaled Dot-Product Attention for each head
attention_outputs = []
scaling_factor = np.sqrt(depth_per_head)
for i in range(num_heads):
    # Compute QK^T
    scores = np.matmul(Q_heads[:, i], K_heads[:, i].transpose(0, 2, 1))  # Shape (1, 2, 2)
    # Scale scores
    scaled_scores = scores / scaling_factor
    # Apply softmax
    attention_weights = softmax(scaled_scores)  # Shape (1, 2, 2)
    # Compute weighted sum of V
    attention_output = np.matmul(attention_weights, V_heads[:, i])  # Shape (1, 2, 2)
    attention_outputs.append(attention_output)

# Step 3: Concatenate heads and project
attention_outputs = np.stack(attention_outputs, axis=1)  # Shape (1, 2, 2, 2)
combined_output = combine_heads(attention_outputs)  # Shape (1, 2, 4)

# Final projection
final_output = np.matmul(combined_output, W)  # Shape (1, 2, 4)
print(final_output)


# //////////////////////////// positional encoding ///////////////////////////////////////

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

# Example: 4 words, embedding size 6
seq_len = 4
d_model = 6
PE = positional_encoding(seq_len, d_model)

print("Positional Encoding:")
print(PE)
