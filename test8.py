import numpy as np
class NN:
    def __init__(self,Nn):
        self.W = 0.10 * np.random.randn(Nn)

    def forward(self, input):
        self.output = np.dot(input, self.W)

    def backward(self, target, inputW):
        self.inW = inputW
        loss = target - self.output
        self.gradient = []
        print(f"loss : {loss}, outout : {self.output}, target : {target}")
        for i in inputW:
            o = loss * i
            self.gradient.append(o)
        #print(self.gradient)
    
    def optim(self, ln=0.01):
        for ind, i in enumerate(self.gradient):
            self.W[ind] = self.W[ind] + ln * i * self.inW[ind]

nn = NN(3)

data = [[1,0,0],[1,1,0],[1,1,1]]
targets = [0.1, 0.5, 1]

for epoch in range(200):
    for ind, batch in enumerate(data):
        nn.forward(batch)
        nn.backward(target=targets[ind], inputW=batch)
        nn.optim()