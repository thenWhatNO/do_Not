import numpy as np
import matplotlib.pyplot as plt

def create_spiral_dataset(n_points, n_classes):
    X = np.zeros((n_points * n_classes, 2))  # data matrix (each row = single example)
    y = np.zeros(n_points * n_classes, dtype='uint8')  # class labels

    for j in range(n_classes):
        ix = range(n_points * j, n_points * (j + 1))
        r = np.linspace(0.0, 1, n_points)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, n_points) + np.random.randn(n_points) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = j

    return X, y

# Parameters
n_points = 100  # number of points per class
n_classes = 2   # number of classes
n_features = 2

# Create dataset
data_x, data_y = create_spiral_dataset(n_points, n_classes)

# Plot dataset
plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, cmap='viridis')
plt.title('Spiral Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
#plt.show()

def one_hot_data(classes, data):
    return np.eye(classes)[data.astype(int)]

def relu(x):
    return np.where(x >=0, x, x*0.01)

def relu_der(x):
    return np.where(x >=0, x, 0.01)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

class NN:
    def __init__(self):
        self.param = [
            [np.random.rand(24, n_features), np.random.rand(1,24),0],
            [np.random.rand(24, 24), np.random.rand(1,24),0],
            [np.random.rand(n_classes, 24), np.random.rand(1,2),1]
        ]
    
    def farward(self, x):
        out = x
        self.param_output = []
        for i in range(len(self.param)):
            Z = np.dot(out, self.param[i][0].T) + self.param[i][1]
            if self.param[i][2] == 1:
                A = sigmoid(Z)
            elif self.param[i][2] == 0:
                A = relu(Z)
            out = A
            self.param_output.append([Z, A])
        return self.param_output
    
    def optim(self ,x , y):
        self.param_updata = []
        d_A = 2 * (self.param_output[2][1] - y)
        for i in range(len(self.param_output)-1, -1, -1):
            if self.param[i][2] == 1:
                d_Z = d_A * sigmoid_der(self.param_output[i][0])
            elif self.param[i][2] == 0:
                d_Z = d_A * relu_der(self.param_output[i][0])
            if i is not 0:
                d_W = np.dot(d_Z.T, self.param_output[i-1][1])
                d_B = np.sum(d_Z, axis=0, keepdims=True) 
            else:
                d_W = np.dot(d_Z.T, x)
                d_B = np.sum(d_Z, axis=0, keepdims=True)
            d_A = np.dot(d_A, self.param[i][0])
            self.param_updata.append([d_W, d_B])
    
    def fit(self, data_x, data_y):
        for i in range(len(data_x)):
            X = [data_x[i]]
            Y = data_y[i]

            self.farward(X)
            self.optim(X, Y)

            for i in range(len(self.param_updata)-1, -1,-1):
                self.oaram.reverse()
                self.param[i][0] -= 0.001 * self.param_updata[i][0]
                self.param[i][1] -= 0.001 * self.param_updata[i][1]

data_y_onehot = one_hot_data(n_classes ,data_y)

model = NN()
model.fit(data_x, data_y_onehot)