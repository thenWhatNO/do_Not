import numpy as np

input = [0.2,0.4,0.6]
W = [0.3,0.6,0.4]
bias = 3

output = input[0] * W[0] + input[1] * W[1] + input[2] * W[2] + bias
print(output)

e = output - 0.8

bias -= e

print(W)

output = input[0] * W[0] + input[1] * W[1] + input[2] * W[2] + bias
print(output)
