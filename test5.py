import numpy as np

data = [[0,0,0,1],[0,0,1,0],
        [0,1,0,0],[1,0,0,0],
        [0,0,1,0],[1,0,0,0],
        [1,0,0,0],[0,0,1,0],
        [0,0,0,1],[0,0,0,1],
        [0,0,1,0],[0,1,0,0],
        [0,0,0,1],[0,1,0,0],
        [0,1,0,0],[1,0,0,0],
        [0,0,1,0],[0,0,0,1],]

targets = [4,3,2,1,3,1,1,3,4,4,3,2,4,2,2,1,3,4]

W1 = np.random.randn(4)
W2 = np.random.randn(1)

def relu(x):
    return np.maximum(0,x)

def der_relu(x):
    return np.where(x>0,1,0)

lerning_R = 0.01

for i in range(1):
    O1 = relu()