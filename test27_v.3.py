import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import sys

###////////////////---activation function---////////////////////

class Relu:
    def run(self, x):
        return np.where(x >= 0, x, 0.001*x)
    
    def derivative(self, x):
        return np.where(x >= 0, x, 0.001)
    

class Sigmoid:
    def run(self, x):
        x = np.array(x)
        if np.any(x > 300) or np.any(x < -300):
            x = np.clip(x, -300, 300) + np.random.normal(0, 0.1, x.shape)
        return  1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        sig = self.run(x)
        return sig * (1 - sig)
    

class swish:
    def run(self, x, beta = 1):
        return x / (1 + np.exp(-beta * x))
    
    def derivative(self, x, beta = 1):
        x = np.array(x)
        sig = 1 / (1 + np.exp(-beta * x))
        return beta * sig * (1 - sig) + sig
    
    
class softmax:
    def run(self, x):
        logits_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)
    
    def derivative(self, x, y):
        out = self.run(x)
        return out - y
    

class Relu:
    def run(self, x):
        return x * np.tanh(np.log1p(np.exp(x)))
    
    def derivative(self, x):
        x = np.array(x)
        omega = 4 * (x - 1) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x)
        delta = 2 * np.exp(x) + np.exp(2 * x) + 2
        return np.exp(x) * omega / (delta ** 2 )



###////////////////---loss function---////////////////////



class BinaryCrossEntropy:
    def loss(self, y_prob, y_targ):
        epsilon = 1e-9 
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        return -np.mean(y_targ * np.log(y_prob) + (1 - y_targ) * np.log(1 - y_prob))

    def derivative(self, y_prob, y_targ):
        epsilon = 1e-9
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        return (y_prob - y_targ) / (y_prob * (1 - y_prob))

    def second_derivative(self, y_prob, y_targ):
        epsilon = 1e-9
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        return (1 - 2 * y_targ) / (y_prob**2 * (1 - y_prob)**2)


class categorical_cross_entropy:
    def loss(self, y_true, y_targ):
        epsilon = 1e-15
        y_targ = np.clip(y_targ, epsilon, 1 - epsilon)  # Clip predictions
        return -np.sum(y_true * np.log(y_targ)) / np.array(y_true).shape[0]
    
    def derivative(self, y_true, y_pred):
        return (np.array(y_pred) - np.array(y_true)).tolist()



###////////////////---leyars---////////////////////



class Dense:
    def __init__(self, input, output, activation_func):
        self.Wight = np.random.randn(output, input) * np.sqrt(2. / input)
        self.Bios = np.random.randn(1, output)
        self.activation_func = activation_func
        self.Z_out = None
        self.A_out = None

    def run(self, X):
        self.Z_out = np.matmul(self.X, self.Wight.T) + self.Bios
        self.A_out = self.activation_func.run(self.Z_out)
        return self.A_out
    
    def optim(self, gradint):
         D_Z = gradint * self.activation_func.derivative(self.Z_out)
         D_W = np.dot(D_Z.T, self.A_out)
         D_B = np.sum(D_Z, axis=0, keepdims=True)
         D_A = np.dot(gradint, self.Wight)

         return [D_W, D_B], D_A
    
    def update_param(self, W, B):
        self.Wight -= W
        self.Bios -= B


class Conv2D:
    def __init__(self, kernel_size, activation_func, filter_num = 1, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel = np.random.randn(filter_num, kernel_size[0], kernel_size[1], 1)
        self.Z_out = None
        self.A_out = None
        self.activation_func = activation_func
        self.input = None

    def run(self, X):

        if self.padding > 0:
            X = np.pad(X, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant').tolist()

        self.input = X

        I_B, I_H, I_W, I_C = X.shape

        O_H = (I_H - np.shape(self.kernel)[1]) // self.stride + 1
        O_W = (I_W - np.shape(self.kernel)[2]) // self.stride + 1
        
        output = np.zeros((I_B, O_H, O_W, np.shape(self.kernel)[0]))

        for y in range(0, O_H):
            for x in range(0, O_W):
                y_start, x_start = y*self.stride, x*self.stride
                y_end, x_end = y_start + np.shape(self.kernel)[1], x_start + np.shape(self.kernel)[2]
                region = X[:, y_start:y_end, x_start:x_end, :]

                output[:, y, x, :] = np.tensordot(region, self.kernel, axes=([1,2,3], [1,2,3]))


    def optim(self, gradint):
        D_A = np.zeros_like(self.input)
        D_K = np.zeros_like(self.kernel)

        for f in range(0, self.kernel[0]):
            for y in range(0, np.shape(gradint)[1]):
                for x in range(0, np.shape(gradint)[2]):
                    for c in range(0, np.shape(gradint)[-1]):
                        y_start, x_start = y*self.stride, x*self.stride
                        y_end, x_end = y_start + np.shape(self.kernel)[1], x_start + np.shape(self.kernel)[2]
                        
                        region = self.input[:, y_start:y_end, x_start:x_end, :]


    def update_param(self):
        pass


###////////////////---opirators---////////////////////
    


class Optim:
    def __init__(self, lerning_rate=0.01):
        self.lern_rate = lerning_rate

        self.time = 1

    def Adam(self, params, beta1=0.9, beta2=0.999, epsilon=1e-8):
        new_poram = []
        for poram in params:
            M = np.zeros_like(poram)
            V = np.zeros_like(poram)

            M = beta1 * M + (1-beta1) * poram
            V = beta2 * V + (1-beta2) * (poram**2)

            hat_M = M / (1-beta1 ** self.time)
            hat_V = V / (1-beta2 ** self.time)

            new_poram.append((self.lern_rate * hat_M / (np.sqrt(hat_V) + epsilon)))
        self.time += 1
        return new_poram
    
    def SVM(self, params):
        new_poram = []
        for poram in params:
            new_poram.append(poram * self.lern_rate)
        return new_poram
    


###////////////////---opiratoNerual network builder---////////////////////



class NN:
    def __init__(self, data, batch, epoch, optimezator, loss):
        self.X_data, self.Y_data = data[0], data[1]
        self.optim = optimezator
        self.loss = loss

        self.batch = batch
        self.epoch = epoch

        self.layers = [
            Dense(2, 2, Relu),
            Dense(2, 4, Relu),
            Dense(4, 1, Sigmoid)
        ]

    def split(self, x_data, y_data, split_procent=0.8):
        split_use = int(len(x_data) * split_use)

        val_data_x = x_data[split_use:]
        val_data_y = y_data[split_use:]
        train_data_x = x_data[:split_use]
        train_data_y = y_data[:split_use]

        return val_data_x, val_data_y, train_data_x, train_data_y

    def farword(self, X):
        data_layer = X
        for layer in self.layers:
            data_layer = layer.run(data_layer)
        return data_layer
    
    def backword(self, gradient):
        grad = gradient
        for layer in reversed(self.layers):
            grad = layer.optim(grad)

    def fit(self):

        val_data_x, val_data_y, train_data_x, train_data_y = self.split(self.X_data, self.Y_data)

        for epoh in range(self.epoch):
            for batch in range(0, len(train_data_x), self.batch):

                x_batch = train_data_x[batch:batch + self.batch]
                y_batch = train_data_y[batch:batch + self.batch]

                output = self.farword(x_batch)

                loss = self.loss.derivative(y_batch, output)
                
                self.backword()
