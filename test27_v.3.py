import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import sys

class Relu:
    def run(self, x):
        return np.where(x >= 0, x, 0.001*x)
    
    def optim(self, x):
        return np.where(x >= 0, x, 0.001)
    
class Sigmoid:
    def run(self, x):
        x = np.array(x)
        if np.any(x > 300) or np.any(x < -300):
            x = np.clip(x, -300, 300) + np.random.normal(0, 0.1, x.shape)
        return  1 / (1 + np.exp(-x))
    
    def optim(self, x):
        sig = self.run(x)
        return sig * (1 - sig)
    
class swish:
    def run(self, x, beta = 1):
        return x / (1 + np.exp(-beta * x))
    
    def optim(self, x, beta = 1):
        x = np.array(x)
        sig = 1 / (1 + np.exp(-beta * x))
        return beta * sig * (1 - sig) + sig

class Dense:
    def __init__(self, input, output, activation_func, X=None, gradint=None):
        self.Wight = np.random.randn(output, input) * np.sqrt(2. / input)
        self.Bios = np.random.randn(1, output)
        self.activation_func = activation_func
        self.Z_out = None
        self.A_out = None
        self.X = X
        self.gradient = gradint

    def run(self):
        self.Z_out = np.matmul(self.X, self.Wight.T) + self.Bios

        self.A_out = self.activation_func.run(self.Z_out)
    
    def optim(self):
         D_Z = self.gradient * self.activation_func.optim(self.Z_out)
         D_W = np.dot(D_Z.T, self.A_out)
         D_B = np.sum(D_Z, axis=0, keepdims=True)

         return [D_W, D_B]
    
class Optim:
    def __init__(self, porameters, lerning_rate=0.01):
        self.params = porameters

        self.time = 1

    def Adam(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        new_poram = []
        for poram in self.params:
            M = np.zeros_like(poram)
            V = np.zeros_like(poram)

            M = beta1 * M + (1-beta1) * poram
            V = beta2 * V + (1-beta2) * (poram**2)

            hat_M = M / (1-beta1 ** self.time)
            hat_V = V / (1-beta2 ** self.time)

            new_poram.append()