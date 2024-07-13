import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Generate the dataset
np.random.seed(0)
data_x = 2 * np.random.rand(50, 1)
data_y = (1-data_x)**2 * 9

plt.scatter(data_x, data_y)
plt.title('Original Data')
plt.show()

# Initialize weights and biases for a network with two hidden layers
W1 = np.random.randn(1, 4) * 0.01
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 4) * 0.01
b2 = np.zeros((1, 4))
W3 = np.random.randn(4, 1) * 0.01
b3 = np.zeros((1, 1))

def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def derv_gelu(x):
    return 0.5 * (1 + erf(x / np.sqrt(2))) + (x * np.exp(-x**2 / 2) / np.sqrt(2 * np.pi))

def forward(x):
    z1 = np.dot(x, W1) + b1
    A1 = gelu(z1)
    
    z2 = np.dot(A1, W2) + b2
    A2 = gelu(z2)
    
    z3 = np.dot(A2, W3) + b3
    A3 = z3  # Linear activation for output layer
    return z1, A1, z2, A2, z3, A3

def backward(X, Y):
    z1, A1, z2, A2, z3, A3 = forward(X)

    d_A3 = 2 * (A3 - Y)
    d_W3 = np.dot(A2.T, d_A3)
    d_b3 = np.sum(d_A3, axis=0, keepdims=True)

    d_A2 = np.dot(d_A3, W3.T)
    d_z2 = d_A2 * derv_gelu(z2)
    d_W2 = np.dot(A1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_A1 = np.dot(d_z2, W2.T)
    d_z1 = d_A1 * derv_gelu(z1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    return d_W1, d_b1, d_W2, d_b2, d_W3, d_b3

learning_rate = 0.01
epochs = 500

# Training the neural network
for epoch in range(epochs):
    for i in range(len(data_x)):
        X = data_x[i].reshape(1, -1)  # Ensure X is a row vector
        Y = data_y[i].reshape(1, -1)  # Ensure Y is a row vector
        
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
    _, _, _, _, _, t_A3 = forward(x.reshape(1, -1))
    predicted_y.append(t_A3[0][0])

plt.scatter(data_x, data_y, label='True Data')
plt.scatter(data_x, predicted_y, label='Predicted Data', color='r')
plt.legend()
plt.title('Model Predictions')
plt.show()
