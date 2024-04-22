import numpy as np
import pandas as pd

data = [[0,1,1,1],
        [0,1,0,0],
        [0,1,1,1],
        [1,1,1,1],
        [0,0,1,1],
        [0,1,1,1],
        [0,0,0,0],
        ]

labels = [1,0,1,1,0,1,0]

input_size = 4
hidden_size = 8
output = 1

Winput_size = np.random.randn(input_size, hidden_size)
biosWin = np.random.randn(1,hidden_size)
Woutput_size = np.random.randn(hidden_size, output)
boisWout = np.random.randn(1, output)

def relu(x):
    return np.maximum(0, x)

def farword(inputs):
    hidden_layer = relu(np.dot(inputs, Winput_size) + biosWin)
    output_layer = relu(np.dot(hidden_layer, Woutput_size) + boisWout)
    return output_layer

def error_count(predictio, target):
    return  np.mean((predictio, target) ** 2)

learning_rate = 0.1

def backward_propogation(input, target, output):
    error = output - target
    output_delta = error * output * (1-output)

    hidden_error = np.dot(output_delta, Winput_size.T)
    hidden_delta = hidden_error * relu(input) * (1 - relu(input))

    Winput_size -= learning_rate * np.dot()