import math
import numpy as np

input = [0.2,0.5,0.7]
W = [0.3,0.3,0.3]

y = np.dot(input, W)
print(y)

targ = 0.5

loss = targ - y

gradient = []
for i in range(3):
    o = loss * input[i]
    gradient.append(o)

print(gradient)

for i in range(3):
    W[i] = W[i] - 0.01 * gradient[i]

print(W)
y = np.dot(input, W)
print(y)