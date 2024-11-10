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
        if self.optim_time:
            self.Dense(25, 1, "sigmoid")
            self.Dense(2, 25, "relu")
            self.optim_time = False
            return
        self.Dense(2, 25, "relu")
        self.Dense(25, 4, "sigmoid")

    def add_output_layer(self, num_layer):
        for i in range(num_layer+1):
            self.Output.append([])
            self.Z_output.append([])
        self.Z_output.pop()

    def Creat(self, ):
        self.optim_time = False
        self.Creat_time = True
        self.Layers()
        self.Creat_time = False

    def Creat_param(self, neruals, weithg):
        np.random.seed(1)
        wight = np.random.randn(neruals, weithg) * np.sqrt(2. / weithg)
        bias = np.random.randn(1, neruals)

        self.Wight.append(wight)
        self.Bias.append(bias) 

    def creat_kernel(self, size):
        kernel = np.random.randn(size, size)

        self.kernel.append(kernel)

    def Activeit(self, Z, activation):
        if activation == "relu": 
            A = derv_relu(Z) if self.optim_time else relu(Z)
        if activation == "sigmoid":
            A = sigmoid_derv(Z) if self.optim_time else sigmoid(Z)
        if activation == "mish":
            A = drev_mish(Z) if self.optim_time else mish(Z)
        if activation == "swish":
            A = drev_swish(Z) if self.optim_time else swish(Z)

        return A 

    def Dense(self, input, output, activition_func):
        if self.Creat_time:
            self.Creat_param(output, input)
            return
        
        if self.optim_time:
            Z_D = self.Output_drev[-1] * self.Activeit(self.Z_output[self.on_this], activition_func)
            self.W_D = np.dot(Z_D.T, self.Output[self.on_this])
            self.B_D = np.sum(Z_D, axis=0, keepdims=True)

            self.Output_drev = np.dot(self.Output_drev[-1], self.Wight[self.on_this])

            self.optim(self.optim_type)
            
            self.on_this -= 1
            return

        Z = np.dot(self.Output[self.on_this][-1], self.Wight[self.on_this].T) + self.Bias[self.on_this]
        self.Z_output[self.on_this].append(Z[0])

        A = self.Activeit(Z, activition_func)
        self.on_this += 1

        self.Output[self.on_this].append(A[0])

    def conv2d(self, kernel_size, stride=1, padding=0):
        if self.Creat_time:
            self.kernel = []
            self.kernel_org_shape = 0
            self.creat_kernel(kernel_size)
            self.conv_optim = False
            return

        input_image = self.Output[self.on_this][-1]

        if self.optim_time:
            kernel_height, kernel_width = self.kernel[self.on_this].shape
            gradient_height, gradient_width = self.Output_drev[self.on_this].shape

            filter_gradient = np.zeros((kernel_height, kernel_width))

            for y in range(0, gradient_height):
                for x in range(0, gradient_width):
                    region = input_image[y:y+kernel_height, x:x+kernel_width]
                    filter_gradient += region * self.Output_drev[self.on_this][y,x]

            self.Output_drev[self.on_this].append(filter_gradient)
            
            self.conv_optim = True
            self.optim(self.optim_type)

            return

        if padding > 0:
            input_image = np.pad(self.Output[self.on_this][-1], ((padding, padding), (padding, padding)), mode='constant')

        input_height, input_width = input_image.shape
        kernel_height, kernel_width = self.kernel[self.on_this].shape

        output_height = (input_height - kernel_height) // stride + 1
        output_width = (input_width - kernel_width) // stride + 1

        output = np.zeros((output_height, output_width))

        for y in range(0, output_height):
            for x in range(0, output_width):
                region = input_image[y*stride:y*stride+kernel_height, x*stride:x*stride+kernel_width]
                opa = region * self.kernel[self.on_this]
                output[y, x] = np.sum(opa)

        self.Output[self.on_this].append(output)
    

    def Flatten(self, img):
        self.kernel_org_shape = np.array(img).shape
        flat = np.array(img).flatten()
        self.Output[self.on_this].append(flat)

    def farword(self, X):
        X = X.reshape(-1, 1).T
        self.on_this = 0

        self.Output[self.on_this].append(X[0])
        self.Layers()
        
    def binary_cros_entropy_drev(self, y_prob, y_targ):
        return y_prob - y_targ

    def optim(self, optimaze_type, lerning_rate = 0.01):

        if optimaze_type == "SVM":
            self.SVM_optim(lerning_rate)
        if optimaze_type == "ADAM":
            self.ADAM_optim(lerning_rate)

    def SVM_optim(self, lerning_rate):
        self.Wight[self.on_this] -= self.W_D * lerning_rate
        self.Bias[self.on_this] -= self.B_D * lerning_rate

    def SVM_optim_conv(self, lerning_rate):
        self.kernel[self.on_this] -= self.Output_drev[self.on_this] * lerning_rate
        
    def ADAM_optim(self, lerning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if not hasattr(self, "time"):
            self.time = 0

        if not hasattr(self, 'm_w'):
            m_w = [np.zeros_like(w) for w in self.Wight]
            v_w = [np.zeros_like(w) for w in self.Wight]
            m_b = [np.zeros_like(b) for b in self.Bias]
            v_b = [np.zeros_like(b) for b in self.Bias]

        m_w[self.on_this] = beta1 * m_w[self.on_this] + (1-beta1) * self.W_D[-1]
        v_w[self.on_this] = beta2 * v_w[self.on_this] + (1-beta2) * (self.W_D[-1]**2)

        hat_m = m_w[self.on_this] / (1-beta1 ** self.time)
        hat_v = v_w[self.on_this] / (1-beta2 ** self.time)

        self.Wight[self.on_this] -= lerning_rate * hat_m / (np.sqrt(hat_v) + epsilon)

        m_b[self.on_this] = beta1 * m_b[self.on_this] + (1-beta1) * self.B_D[-1]
        v_b[self.on_this] = beta2 * v_b[self.on_this] + (1-beta2) * (self.B_D[-1]**2)

        bhat_m = m_b[self.on_this] / (1-beta1 ** self.time)
        bhat_v = v_b[self.on_this] / (1-beta2 ** self.time)

        self.Bias[self.on_this] -= lerning_rate * bhat_m / (np.sqrt(bhat_v) + epsilon)
                
        self.time += 1

    def fit(self, epoch, batch_size, optimizer_name):
        self.optim_type = optimizer_name

        for i in range(epoch):
            for point in range(0, len(self.X_data), batch_size):
                self.add_output_layer(2)

                x_batch = self.X_data[point:point + batch_size]
                self.y_batch = self.Y_data[point:point + batch_size]

                for unit in x_batch:
                    output = self.farword(unit)

                self.batch_label = self.Output[-1]
 
                self.optim_time = True
                self.Output_drev = []
                A_drev = self.binary_cros_entropy_drev(self.batch_label, self.y_batch)
                self.Output_drev.append(A_drev)

                self.on_this -= 1
                self.Layers()

model = NN(data_set1)
model.Creat()
model.fit(1, 10, 'ADAM')