import math
import numpy as np

input = [[1,0,0],[1,1,0],[1,1,1]]
W = [0.3,0.3,0.3]
bias = 1

targ = [0.1, 0.5, 1]

for i in range(30):
    R = np.random.randint(0,3)

    y = np.dot(input[R], W)
    
    loss = targ[R] - y
    
    gradient = []
    for i in range(3):
        o = targ[R] - np.abs(input[R][i])
        gradient.append(o)
    
    gradient2 = np.dot(loss,input[R])

    bias = bias + 0.01 * loss

    print(f"grad 1 : {gradient} grad 2 {gradient2}")

    for i in range(3):
        W[i] = W[i] + 0.01 * gradient[i] * input[R][i]
    print(f"loss : {loss}, target : {targ[R]}, output N : {y}")