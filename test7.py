import math
import numpy as np
import matplotlib.pyplot as plt

input = [[1,0,0],[1,1,0],[1,1,1]]
W = [0.3,0.3,0.3]
bias = 1

targ = [0.1, 0.5, 1]

loss_plot = []
i_plot = []


for i in range(300):

    R = np.random.randint(0,3)

    y = np.dot(input[R], W)
    
    loss = np.abs(targ[R] - y)
    
    gradient = []
    for ii in range(3):
        o = targ[R] - np.abs(input[R][ii])
        gradient.append(o)
    
    gradient2 = np.dot(loss,input[R])

    bias = bias + 0.001 * loss

    for r in range(3):
        W[r] = W[r] + 0.001 * gradient[r] * input[R][r]
    print(f"loss : {loss}, target : {targ[R]}, output N : {y}")

    loss_plot.append(loss)
    i_plot.append(i)
    plt.scatter(i, y, color='red')

plt.plot(i_plot, loss_plot, color='orange')
plt.show()