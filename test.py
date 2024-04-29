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
hidden_size = 4
output = 1

Winput_size = np.random.randn(hidden_size, input_size)
biosWin = np.random.randn(1,hidden_size)
Woutput_size = np.random.randn(output, hidden_size)
boisWout = np.random.randn(1, output)

#print(Winput_size.shape, '\n',Woutput_size.shape)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def farword(inputs):
    hidden_layer = sigmoid(np.dot(Winput_size, inputs))
    print(hidden_layer.shape,'\n\n')
    output_layer = sigmoid(np.dot(Woutput_size, hidden_layer))
    print(output_layer.shape,'\n\n')
    return hidden_layer ,output_layer

def error_count(predictio, target):
    return np.mean((predictio, target) ** 2)

def backward_propogation(input, target):
    global Woutput_size, Winput_size, biosWin, boisWout
    learning_rate = 0.01
    N = 1

    for k in range(N):
        get = np.random.randint(0,7)
        x = input[get]
        y, out = farword(x)
        e = y - target[get]

        #print(f"data -> {x}, output -> {out}, error -> {e}, target -> {target[get]}")

        gradient = e * out * (1 - out)

        print(Woutput_size.shape, (learning_rate * gradient * out).shape)
        Woutput_size = Woutput_size - learning_rate * gradient * out

        gradient1 = Woutput_size * gradient * (1 - y)

        Winput_size = Winput_size - learning_rate * gradient1 * y

#farword(data[0])
backward_propogation(data,labels)