import numpy as np

input = [0.2,0.4,0.6]
W = [[0.3,0.6,0.4],
     [0.2,0.5,0.2]]
bias = 3

class NN:
    def __init__(self, inp, nnn):
        self.W1 = 0.10 * np.random.randn(inp, nnn)
        self.bais = np.zeros(1, nnn)
    def farward(self, inputs):
        self.output = np.dot(input, self.W1) + self.bais
        
NN1 = NN(3, 6)
NN2 = NN(6, 2)

NN1.farward(input)
print(NN1.output)