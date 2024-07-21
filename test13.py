import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

np.random.seed(0)
data_x = 2 * np.random.rand(50, 1)
data_y = np.sin(data_x)

plt.scatter(data_x, data_y)

W1 = np.random.randn(25, 1)
b1 = np.random.randn(1, 25)
W2 = np.random.randn(25, 25)
b2 = np.random.randn(1, 25)
W3 = np.random.randn(1, 25)
b3 = np.random.randn(1, 1)

def relu(x):
    return np.where(x >= 0, x, 0.001*x)
def derv_relu(x):
    return np.where(x >= 0, x, 0.001)

def farward(x):
    z1 = np.dot(x, W1.T) + b1
    A1 = relu(z1)

    z2 = np.dot(A1, W2.T) + b2
    A2 = relu(z2)

    z3 = np.dot(A2, W3.T) + b3
    return z1, A1, z2, A2, z3

def back(X, Y):
    z1, A1, z2, A2, z3 = farward(X)

    d_A3 = 2 * (z3 - Y)
    d_W3 = np.dot(d_A3, A2)
    d_b3 = np.sum(d_A3, axis=0, keepdims=True)

    d_A2 = np.dot(d_A3, W3)
    d_z2 = d_A2 * derv_relu(z2)
    d_W2 = np.dot(d_z2.T, A1)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_A1 = np.dot(d_A2, W2)
    d_z1 = d_A1 * derv_relu(z1)
    d_W1 = np.dot(d_z1.T, X)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    return d_W1, d_b1, d_W2, d_b2, d_W3, d_b3

for i in range(3000):
    for i in range(len(data_x)):

        X = data_x[i].reshape(1, -1) 
        Y = data_y[i].reshape(1, -1)

        d_W1 , d_b1, d_W2, d_b2, d_W3, d_b3  = back(X, Y)

        W1 -= 0.001 * d_W1
        b1 -= 0.001 * d_b1
        W2 -= 0.001 * d_W2 
        b2 -= 0.001 * d_b2
        W3 -= 0.001 * d_W3
        b3 -= 0.001 * d_b3

for ind, x in enumerate(data_x):
    t_z1 = np.dot(x, W1.T) + b1
    t_A1 = relu(t_z1)

    t_z2 = np.dot(t_A1, W2.T) + b2
    t_A2 = relu(t_z2) 

    t_z3 = np.dot(t_A2, W3.T)

    plt.scatter(x, t_z3, color="red")

plt.show()
