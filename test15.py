import numpy as np
import matplotlib.pyplot as plt

# Generate a dataset with two clusters for binary classification
np.random.seed(0)
n_samples_per_cluster = 50
n_features = 2
n_clusters = 3

# Means and standard deviations for the clusters
means = [(-2, -2), (2, 2), (-3, 5)]
std_devs = [0.5, 0.5, 0.5]

# Generate random data for each cluster
data_x = np.zeros((n_samples_per_cluster * n_clusters, n_features))
data_y = np.zeros(n_samples_per_cluster * n_clusters)

for i in range(n_clusters):
    data_x[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster] = np.random.normal(
        loc=means[i], scale=std_devs[i], size=(n_samples_per_cluster, n_features))
    data_y[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster] = i

print(data_y)

# Visualize the dataset
plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, cmap='viridis')
plt.title('Two Cluster Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Initialize weights and biases for a network with two hidden layers and sigmoid output layer
W1 = np.random.randn(4, n_features) * 0.01
b1 = np.zeros((4, 1))
W2 = np.random.randn(4, 4) * 0.01
b2 = np.zeros((4, 1))
W3 = np.random.randn(1, 4) * 0.01
b3 = np.zeros((1, 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def derv_gelu(x):
    tanh_out = np.tanh(0.797885 * x + 0.035677 * np.power(x, 3))
    return 0.5 * (1 + tanh_out) + 0.5 * x * (1 - tanh_out ** 2) * (0.797885 + 0.107032 * np.power(x, 2))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derv_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def forward(x):
    z1 = np.dot(W1, x) + b1
    A1 = gelu(z1)
    
    z2 = np.dot(W2, A1) + b2
    A2 = gelu(z2)
    
    z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(z3)
    return z1, A1, z2, A2, z3, A3

def backward(X, Y):
    z1, A1, z2, A2, z3, A3 = forward(X)

    d_A3 = 2 * (A3 - Y) * derv_sigmoid(z3)
    d_W3 = np.dot(d_A3, A2.T)
    d_b3 = np.sum(d_A3, axis=1, keepdims=True)

    d_A2 = np.dot(W3.T, d_A3)
    d_z2 = d_A2 * derv_gelu(z2)
    d_W2 = np.dot(d_z2, A1.T)
    d_b2 = np.sum(d_z2, axis=1, keepdims=True)

    d_A1 = np.dot(W2.T, d_z2)
    d_z1 = d_A1 * derv_gelu(z1)
    d_W1 = np.dot(d_z1, X.T)
    d_b1 = np.sum(d_z1, axis=1, keepdims=True)

    return d_W1, d_b1, d_W2, d_b2, d_W3, d_b3

learning_rate = 0.01
epochs = 500

# Training the neural network
for epoch in range(epochs):
    for i in range(len(data_x)):
        X = data_x[i].reshape(-1, 1)  # Ensure X is a column vector
        Y = data_y[i].reshape(-1, 1)  # Ensure Y is a column vector
        
        d_W1, d_b1, d_W2, d_b2, d_W3, d_b3 = backward(X, Y)

        W1 -= learning_rate * d_W1
        b1 -= learning_rate * d_b1
        W2 -= learning_rate * d_W2
        b2 -= learning_rate * d_b2
        W3 -= learning_rate * d_W3
        b3 -= learning_rate * d_b3

# Visualizing the model predictions
predicted_y = []
for x in data_x:
    _, _, _, _, _, t_A3 = forward(x.reshape(-1, 1))
    predicted_y.append(t_A3[0][0])

predicted_y = np.array(predicted_y)
predicted_labels = (predicted_y > 0.5).astype(int)

plt.scatter(data_x[:, 0], data_x[:, 1], c=predicted_labels, cmap='viridis', marker='x', label='Predicted Data')
plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, cmap='viridis', marker='o', alpha=0.5, label='True Data')
plt.legend()
plt.title('Model Predictions')
plt.show()
