import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
data_x = 2 * np.random.rand(50, 1)
data_y = 4 + 3 * data_x**3 + np.random.randn(50,1)**2

plt.scatter(data_x, data_y)
plt.show()

W1 = np.array([[0.1],
               [0.1],
               [0.1]])
b1 = np.array([0.1,0.1,0.1])
W2 = np.array([[0.1,0.1,0.1]])
b2 = np.array([0.1])

def relu(X):
    return np.maximum(0, X)

def derv_relu(x):
    return np.where(x > 0,1,0)

def farward(x):
    z1 = np.dot(x, W1.T) + b1
    A1 = relu(z1)

    z2 = np.dot(A1, W2.T) + b2
    A2 = z2
    return z1, A1, z2, A2

def back(X, Y):
    z1, A1, z2, A2 = farward(X[i])

    d_A2 = 2 * (A2 - Y[i])
    d_W2 = np.dot(d_A2, A1)
    d_b2 = np.sum(d_A2, axis=0, keepdims=True)

    d_A1 = np.dot(W2.T, d_A2)
    d_z1 = d_A1 .T* derv_relu(z1)
    d_W1 = np.dot(d_z1, X[i])
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    return d_W1, d_b1, d_W2, d_b2

for i in range(10):
    for i in range(len(data_x)):
        d_W1 , d_b1, d_W2, d_b2 = back(data_x[i], data_y[i])

        W1 -= 0.01 * d_W1
        b1 -= 0.01 * d_b1
        W2 -= 0.01 * d_W2 
        b2 -= 0.01 * d_b2

for ind, x in enumerate(data_x):
    t_z1 = np.dot(x, W1.T) + b1
    t_A1 = relu(t_z1)

    t_z2 = np.dot(t_A1, W2) + b2
    t_A2 = t_z2

    plt.scatter(t_A2, data_y[ind])

plt.show()