import numpy as np

input = [0.2,0.4,0.6]
W = [[0.3,0.6,0.4],
     [0.2,0.5,0.2]]
bias = 3

class NN:
    def __init__(self, inp, nnn):
        self.W1 = 0.10 * np.random.randn(inp, nnn)
        self.bais = np.zeros((1, nnn))
    def farward(self, inputs):
        self.output = np.dot(inputs, self.W1) + self.bais

class Relu:
    def farward(self, x):
        self.output = np.maximum(0, x)
class softmax:
    def farward(self, x):
        exp_val = np.exp(x - np.max(x, axis=1, keepdims=True))
        prababilty = exp_val /np.sum(exp_val, axis=1, keepdims=True)
        self.output = prababilty

class loss:
    def counting(self, output, y):
        simply_loss = self.farward(output, y)
        data_loss = np.mean(simply_loss)
        return data_loss
    
class loss_cat(loss):
    def farward(self, y, targ):
        simplis = len(y)
        y_clip = np.clip(y, 1e-7, 1-1e-1)

        if len(y) == 1:
            cor = y_clip[range(simplis), targ]
        elif len(y) ==2:
            cor = np.sum(y_clip * targ, axis=1)
        negcor = -np.log(cor)
        return negcor

        
NN1 = NN(3, 6)
NN2 = NN(6, 2)

loss_func = loss_cat()
losss = loss_func.counting([[5,5]], [0,0])

print(losss)