import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
data_x = 2 * np.random.rand(50, 1)
data_y = -(1-data_x)**2 * 5 + 2

plt.scatter(data_x, data_y)
plt.title('Original Data')
plt.show()

# Initialize weights and biases
W1 = np.random.randn(1, 6)
b1 = np.random.randn(1, 6)
W2 = np.random.randn(6, 1)
b2 = np.random.randn(1, 1)

def relu(X):
    return np.maximum(0, X)

def derv_relu(x):
    return np.where(x > 0, 1, 0)

def forward(x):
    z1 = np.dot(x, W1) + b1
    A1 = relu(z1)
    z2 = np.dot(A1, W2) + b2
    A2 = z2  # Linear activation for output layer
    return z1, A1, z2, A2

def backward(X, Y):
    z1, A1, z2, A2 = forward(X)

    d_A2 = 2 * (A2 - Y)
    d_W2 = np.dot(A1.T, d_A2)
    d_b2 = np.sum(d_A2, axis=0, keepdims=True)

    d_A1 = np.dot(d_A2, W2.T)
    d_z1 = d_A1 * derv_relu(z1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    return d_W1, d_b1, d_W2, d_b2

learning_rate = 0.01
epochs = 100

# Training the neural network
for epoch in range(epochs):
    for i in range(len(data_x)):
        X = data_x[i].reshape(1, -1)  # Ensure X is a row vector
        Y = data_y[i].reshape(1, -1)  # Ensure Y is a row vector
        
        d_W1, d_b1, d_W2, d_b2 = backward(X, Y)

        W1 -= learning_rate * d_W1
        b1 -= learning_rate * d_b1
        W2 -= learning_rate * d_W2
        b2 -= learning_rate * d_b2

# Visualizing the model predictions
predicted_y = []
for x in data_x:
    _, _, _, t_A2 = forward(x.reshape(1, -1))
    predicted_y.append(t_A2[0][0])

plt.scatter(data_x, data_y, label='True Data')
plt.scatter(data_x, predicted_y, label='Predicted Data', color='r')
plt.legend()
plt.title('Model Predictions')
plt.show()
