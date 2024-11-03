import numpy as np
import matplotlib.pyplot as plt

#################### the grid

x_min, x_max, y_min, y_max = -3.5, 3.5, -3.5, 3.5
step_size = 0.25

x_values = np.arange(x_min, x_max, step_size)
y_values = np.arange(y_min, y_max, step_size)
x_grid, y_grid = np.meshgrid(x_values, y_values)

x_point = x_grid.flatten()
y_point = y_grid.flatten()

#///////////// other data base

n_samples_per_cluster = 50  
n_features = 2        
n_clusters = 4

means = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
std_devs = [0.5, 0.5, 0.5, 0.5]

data_x = np.zeros((n_samples_per_cluster * n_clusters, n_features))
data_y = np.zeros(n_samples_per_cluster * n_clusters)

for i in range(n_clusters):
    data_x[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster] = np.random.normal(
        loc=means[i], scale=std_devs[i], size=(n_samples_per_cluster, n_features))
    data_y[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster] = i

def one_how(labels, num_class):
    return np.eye(num_class)[labels.astype(int)]

data_y_one_hot = one_how(data_y, n_clusters)

plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, cmap='viridis')
plt.show()

data_set1 = [data_x, data_y_one_hot]


###################### the other function #########################



def sigmoid(x):
    x = np.array(x)
    if np.any(x > 300) or np.any(x < -300):
        x = np.clip(x, -300, 300) + np.random.normal(0, 0.1, x.shape)
    return  1 / (1 + np.exp(-x))

def sigmoid_derv(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    return np.where(x >= 0, x, 0.001*x)

def derv_relu(x):
    x = np.array(x)
    return np.where(x >= 0, x, 0.001)

def swish(x, beta = 1):
    return x / (1 + np.exp(-beta * x))

def drev_swish(x, beta = 1):
    x = np.array(x)
    sig = 1 / (1 + np.exp(-beta * x))
    return beta * sig * (1 - sig) + sig

def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))

def drev_mish(x):
    x = np.array(x)
    omega = 4 * (x - 1) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x)
    delta = 2 * np.exp(x) + np.exp(2 * x) + 2
    return np.exp(x) * omega / (delta ** 2 )

class NN:
    def __init__(self, data):
        self.X_data, self.Y_data = data[0], data[1]

        self.Wight = []
        self.Bias = []

        self.Output = []
        self.Z_output = []

    def Layers(self):   
        self.Dense(2, 25, "relu")
        self.Dense(25, 1, "sigmoid")

    def add_output_layer(self, num_layer):
        for i in range(num_layer+1):
            self.Output.append([])
            self.Z_output.append([])
        self.Z_output.pop()

    def Creat(self, ):
        self.Creat_time = True
        self.Layers()
        self.Creat_time = False

    def Creat_param(self, neruals, weithg):
        np.random.seed(1)
        wight = np.random.randn(neruals, weithg) * np.sqrt(2. / weithg)
        bias = np.random.randn(1, neruals)

        self.wight.append(wight)
        self.bias.append(bias) 

    def Activeit(self, Z, activation):
        if activation == "relu":
            A = relu(Z)
        if activation == "sigmoid":
            A = sigmoid(Z)
        if activation == "mish":
            A = mish(Z)
        if activation == "swish":
            A = swish(Z)

        return A 

    def Dense(self, input, output, activition_func):
        if self.Creat_time:
            self.Creat_param(output, input)
            return

        Z = np.dot(self.Output[self.on_this][-1], self.Wight.T) + self.Bias
        self.Z_output.append(Z)

        A = self.Activeit(Z, activition_func)
        self.Output.append(A)
        
        self.on_this += 1

    def farword(self, X):
        X = X.reshape(-1, 1).T
        self.on_this = 0

        self.Output.append(X)
        self.Layers()

    def fit(self, epoch, batch_size):

        for i in range(epoch):
            for point in range(0, len(self.X_data), batch_size):
                self.add_output_layer(2)

                x_batch = self.X_data[point + batch_size]
                y_batch = self.Y_data[point + batch_size]

                self.batch_label = []
                
                for unit in x_batch:
                    output = self.farword(unit)
                    self.batch_label.append(output)