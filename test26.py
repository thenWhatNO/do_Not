import numpy as np

# Inputs
X = np.array([
    [2, 2, 2, 2],
    [2, 2, 2, 2]
])  # shape (2, 4)

grad_output = np.array([
    [1., 1., 1., 1.],
    [1., 1., 1., 1.]
])  # shape (2, 4)

# Weights
V_w = np.array([
    [1, 2, 2, 1],
    [1, 1, 2, 1],
    [2, 2, 1, 1],
    [2, 2, 2, 2]
])  # shape (4, 4)

Q_w = np.array([
    [2, 1, 2, 2],
    [2, 1, 2, 2],
    [1, 1, 1, 2],
    [2, 1, 1, 2]
])  # shape (4, 4)

K_w = np.array([
    [1, 2, 2, 2],
    [2, 1, 1, 2],
    [2, 2, 2, 2],
    [1, 1, 2, 2]
])  # shape (4, 4)

O_w = np.array([
    [2, 1, 2, 1],
    [1, 1, 2, 2],
    [2, 1, 2, 1],
    [1, 2, 2, 2]
])  # shape (4, 4)

# Forward step
Q = X @ Q_w  # (2, 4)
K = X @ K_w  # (2, 4)
V = X @ V_w  # (2, 4)

# Attention scores
scores = Q @ K.T / np.sqrt(Q.shape[-1])  # scaled dot product
softmax_scores = np.exp(scores)
softmax_scores /= np.sum(softmax_scores, axis=-1, keepdims=True)

# Output of attention
attn_output = softmax_scores @ V  # (2, 4)

# Final output
out = attn_output @ O_w  # (2, 4)

# Backward step
# Gradient wrt O_w
dO_w = attn_output.T @ grad_output  # (4, 4)

# Gradient wrt attention output
d_attn_output = grad_output @ O_w.T  # (2, 4)

# Gradient wrt softmax_scores
d_softmax = d_attn_output @ V.T  # (2, 2)

# Jacobian of softmax
def softmax_derivative(s):
    s = s.reshape(-1, 1)
    return np.diagflat(s) - s @ s.T

# Compute gradients wrt Q, K, V
dV = softmax_scores.T @ d_attn_output  # (2, 4)

d_scores = np.empty_like(scores)
for i in range(scores.shape[0]):
    J = softmax_derivative(softmax_scores[i])
    d_scores[i] = J @ d_softmax[i]


dQ = d_scores @ K  # (2, 4)
dK = d_scores.T @ Q  # (2, 4)

# Gradient wrt Q_w, K_w, V_w
dQ_w = X.T @ dQ  # (4, 4)
dK_w = X.T @ dK  # (4, 4)
dV_w = X.T @ dV  # (4, 4)

dQ.shape, dK.shape, dV.shape, dQ_w.shape, dK_w.shape, dV_w.shape, dO_w.shape
print("help[me]")