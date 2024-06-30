import numpy as np
import matplotlib.pylab as plt

np.random.seed(0)
data_x = 3 * np.random.rand(50, 1)
data_y = 1 + data_x**4 + np.random.rand(50, 1)**2

W1 = np.array([0.1])
b1 = np.array([0.1])
W2 = np.array([[0.1],[0.1]])
b2 = np.array([0.1,0.1])
outW = np.array([0.1,0.1])
outb = np.array([0.1])

def relu(X):
    return np.maximum(0, X)

def derv_relu(x):
    return np.where(x > 0,1,0)

def farword(w1,b1,w2,b2,outw,outb, input):
    layar_out1 = np.dot(input, w1) + b1
    activ_func1 = relu(layar_out1)
    layar_out2 = np.dot(activ_func1, w2.T) + b2
    activ_func2 = relu(layar_out2)
    output_layer = np.dot(activ_func2, outw) + outb
    return activ_func1, activ_func2, output_layer

def optim(y, ln=0.01):
    out1, out2, y_hat = farword(W1,b1,W2,b2,outW,outb,data_x[0][0])
    error = y - y_hat
    out_grad = error * derv_relu(y)
    W2_grad = outW * out_grad * derv_relu(out2)
    W1_grad = W2 * W2_grad * derv_relu(out1)
    grad = W1 * W1_grad * derv_relu(data_x[0][0])

    print(f"out_grad {out_grad}, W2_grad { W2_grad}, W1_grad{W1_grad}, grad {grad}")

optim(data_y[0][0])