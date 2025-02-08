import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # Prevent overflow
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def softmax_derivative(softmax_output):
    return softmax_output * (1 - softmax_output)  # Element-wise derivative

class Transformer:
    def __init__(self, d_model, head_num):
        self.d_model = d_model
        self.head_num = head_num
        self.depth = d_model // head_num  # Depth per head

        # Initialize weight matrices for all heads
        self.W_Q = np.random.randn(d_model, d_model)
        self.W_K = np.random.randn(d_model, d_model)
        self.W_V = np.random.randn(d_model, d_model)
        self.W_O = np.random.randn(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Splits the input tensor into multiple heads.
        """
        x = x.reshape(batch_size, -1, self.head_num, self.depth)  # (batch, seq, head_num, depth)
        return x.transpose(0, 2, 1, 3)  # (batch, head_num, seq, depth)

    def combine_heads(self, x):
        """
        Combines multiple attention heads into a single matrix.
        """
        x = x.transpose(0, 2, 1, 3)  # (batch, seq, head_num, depth)
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.d_model)  # Merge heads back

    def multi_head_attention(self, Q, K, V, grad=None):
        """
        Implements Multi-Head Self-Attention with forward and backward pass.
        """
        batch_size = Q.shape[0]

        # Forward Pass
        Q = Q @ self.W_Q
        K = K @ self.W_K
        V = V @ self.W_V

        Q_heads = self.split_heads(Q, batch_size)
        K_heads = self.split_heads(K, batch_size)
        V_heads = self.split_heads(V, batch_size)

        scores = Q_heads @ K_heads.transpose(0, 1, 3, 2) / np.sqrt(self.depth)  # (batch, head_num, seq, seq)
        attention_weights = softmax(scores, axis=-1)  # (batch, head_num, seq, seq)
        O_heads = attention_weights @ V_heads  # (batch, head_num, seq, depth)

        O = self.combine_heads(O_heads) @ self.W_O  # Final projection

        # Backward Pass (if training)
        if grad is not None:
            dO = grad @ self.W_O.T  # Gradient w.r.t. combined output
            dO_heads = self.split_heads(dO, batch_size)  # Split into heads again

            dV_heads = attention_weights.transpose(0, 1, 3, 2) @ dO_heads  # dV
            d_attention_weights = dO_heads @ V_heads.transpose(0, 1, 3, 2)  # d_attention_weights

            d_scores = d_attention_weights * softmax_derivative(attention_weights)  # Apply derivative
            dQ_heads = d_scores @ K_heads  # dQ
            dK_heads = d_scores.transpose(0, 1, 3, 2) @ Q_heads  # dK

            # Combine gradients back
            dQ = self.combine_heads(dQ_heads)
            dK = self.combine_heads(dK_heads)
            dV = self.combine_heads(dV_heads)

            dW_Q = Q.T @ dQ
            dW_K = K.T @ dK
            dW_V = V.T @ dV
            dW_O = O.T @ grad  # Update final projection weight

            return O, (dW_Q, dW_K, dW_V, dW_O)  # Return gradients for updating weights

        return O  # Just return output in forward mode

# Example Usage
batch_size = 2
seq_len = 5
d_model = 8
head_num = 2

transformer = Transformer(d_model, head_num)
Q = np.random.randn(batch_size, seq_len, d_model)
K = np.random.randn(batch_size, seq_len, d_model)
V = np.random.randn(batch_size, seq_len, d_model)

O, gradients = transformer.multi_head_attention(Q, K, V, grad=np.random.randn(batch_size, seq_len, d_model))

print("Output shape:", O.shape)
print("Gradients shape (W_Q, W_K, W_V, W_O):", [g.shape for g in gradients])
