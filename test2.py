import numpy as np
import pandas as pd

data_path = "plenet_data.csv"
df = pd.read_csv(data_path)

Y = np.array(df['1'])
X = np.array(df.drop('1', axis=1))

W_input_layer = np.random.randn(1,5)
W_hidden_layer_1 = np.random.randn(5,5)
W_hidden_layer_2 = np.random.randn(5,1)
W_output_layer = np.random.randn(5,1)

print(W_input_layer.shape, '\n\n',W_hidden_layer_1.shape, '\n\n',W_hidden_layer_2.shape, '\n\n',W_output_layer.shape)

bias_1 = np.random.randn(1, 1)
bias_2 = np.random.randn(1, 1)
bias_3 = np.random.randn(1, 1)
bias_4 = np.random.randn(1, 1)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def farword(x):
    out_1 = np.dot(x, W_input_layer) + bias_1
    in_1 = relu(out_1)
    out_2 = np.dot(in_1, W_hidden_layer_1) + bias_2
    in_2 = relu(out_2)
    out_3 = np.dot(in_2, W_hidden_layer_2) + bias_3
    in_3 = relu(out_3)
    out_4 = np.dot(in_3.T, W_output_layer) + bias_4
    in_4 = relu(out_4)

    #print(out_1.shape, out_2.shape, out_3.shape, out_4.shape)
    return out_1, out_2, out_3, in_4

def backword(data, labels):
    global W_input_layer, W_hidden_layer_1, W_hidden_layer_2, W_output_layer
    global bias_1, bias_2, bias_3, bias_4
    learning_rate = 0.01
    N = 1000

    for k in range(N):
        num = np.random.randint(0,13)
        x = data[num]
        out1, out2, out3, out4 = farword(np.array([x]).T)
        E = out4-(labels[num])

        print(f'output : {out4}, error : {E}, target : {labels[num]}')

        gradient_1 = E * out4 * (1-out4)
        W_output_layer = W_output_layer - learning_rate * gradient_1 * out4
        bias_4 = bias_4 - learning_rate * np.sum(gradient_1, axis=0, keepdims=True)

        gradient_2 = W_output_layer * gradient_1 * (1-out3)
        W_hidden_layer_2 = W_hidden_layer_2 - learning_rate * gradient_2 * out3
        bias_3 = bias_3 - learning_rate * np.sum(gradient_2, axis=0, keepdims=True)

        gradient_3 = W_hidden_layer_2 - learning_rate * (1-out2)
        W_hidden_layer_1 = W_hidden_layer_1 - learning_rate * gradient_3 * out2
        bias_2 = bias_2 - learning_rate * np.sum(gradient_3, axis=0, keepdims=True)


backword(X, Y)