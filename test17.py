import numpy as np
import matplotlib.pylab as plt

#################### the grid

x_min, x_max, y_min, y_max = -1.5, 1.5, -1.5, 1.5
step_size = 0.1

x_values = np.arange(x_min, x_max, step_size)
y_values = np.arange(y_min, y_max, step_size)
x_grid, y_grid = np.meshgrid(x_values, y_values)

x_point = x_grid.flatten()
y_point = y_grid.flatten()

plt.scatter(x_point, y_point, color="black", marker='o')

#/////////////spiral data base

np.random.seed(1)

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

n_points = 100
n_classes = 2   
n_features = 2

data_x, data_y = create_spiral_dataset(n_points, n_classes)

def one_how(labels, num_class):
    return np.eye(num_class)[labels.astype(int)]

one_shot_dataY = one_how(data_y, n_classes)

plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, cmap='viridis')
plt.title('Spiral Dataset')
plt.show()

data_set = [[data_x], one_shot_dataY]


#///////////// other data base

# n_samples_per_cluster = 50  
# n_features = 2        
# n_clusters = 4

# means = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
# std_devs = [0.5, 0.5, 0.5, 0.5]

# data_x = np.zeros((n_samples_per_cluster * n_clusters, n_features))
# data_y = np.zeros(n_samples_per_cluster * n_clusters)

# for i in range(n_clusters):
#     data_x[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster] = np.random.normal(
#         loc=means[i], scale=std_devs[i], size=(n_samples_per_cluster, n_features))
#     data_y[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster] = i

# def one_how(labels, num_class):
#     return np.eye(num_class)[labels.astype(int)]

# data_y_one_hot = one_how(data_y, n_clusters)

# # plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, cmap='viridis')
# # plt.show()

# data_set1 = [[data_x], data_y_one_hot]


###################### the other function #########################
def sigmoid(x):
    if np.any(x > 300) or np.any(x < -300):
        x = np.clip(x, -300, 300) + np.random.normal(0, 0.1, x.shape)
    return  1 / (1 + np.exp(-x))

def sigmoid_derv(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    return np.where(x >= 0, x, 0.001*x)

def derv_relu(x):
    return np.where(x >= 0, x, 0.001)

class NN:
    def __init__(self, data):
        self.X_data, self.Y_data = data[0], data[1]
        self.wight = []
        self.bias = []
        self.A_output = []
        self.Z_output = []
        self.optin_time = False

        self.d_a = []
        self.d_z = []
        self.d_wight = []
        self.d_bias = []

        self.layers = [
            (self.relu, 2, 25),
            (self.relu, 25, 30),
            (self.relu, 30, 45),
            (self.relu, 45, 25),
            (self.relu, 25, 25),
            (self.sigmoid, 25, 2)
        ]

    def CreatLayers(self):
        self.creat = True
        for func, arg1, arg2 in self.layers:
            func(arg1, arg2)
        self.creat = False

    def creatParam(self, weithg, neruals):
        np.random.seed(1)
        wight = np.random.randn(neruals, weithg) * np.sqrt(2. / weithg)
        bias = np.random.randn(1, neruals)

        self.wight.append(wight)
        self.bias.append(bias)

    def relu(self, input, output):
        if(self.creat):
            self.creatParam(input, output)
        else:
            if(self.optin_time):
                Da = self.d_a[-1] * derv_relu(self.Z_output[self.stack_func_opiration])
                self.d_z.append(Da)
            else:
                A = relu(self.Z_output[-1])
                self.A_output.append(A)

    def sigmoid(self, input, output):
        if(self.creat):
            self.creatParam(input, output)
        else:
            if(self.optin_time):
                D_A_OUT = self.d_a[-1] * sigmoid_derv(self.Z_output[self.stack_func_opiration])
                self.d_z.append(D_A_OUT)
            else:
                A = sigmoid(self.Z_output[-1])
                self.A_output.append(A)

    def farword(self, X):
        self.stack_opiration = [X.reshape(-1, 1).T]
        self.stack_func_opiration = 0

        for layer, arg1, arg2 in self.layers:
            Z = np.dot(self.stack_opiration[-1], self.wight[self.stack_func_opiration].T) + self.bias[self.stack_func_opiration]
            self.Z_output.append(Z)

            layer(arg1, arg2)
            self.stack_opiration.append(self.A_output[-1])
            self.stack_func_opiration += 1

    def binary_cros_entropy(self, y_prob, y_targ):
        epsilon = 1e-8
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        return -np.sum(y_targ * np.log(y_prob))
    
    def binary_cros_entropy_drev(self, y_prob, y_targ):
        return y_prob - y_targ

    def optim(self, Y):
        self.optin_time = True
        self.Derev_opiration = 0
        d_a3 = 2 * (self.stack_opiration[-1] - Y)
        self.d_a.append(d_a3)

        for layer, arg1, arg2 in reversed(self.layers):
            self.stack_func_opiration -= 1
            layer(arg1, arg2)

            self.d_wight.append(np.dot(self.d_z[-1].T, self.stack_opiration[self.stack_func_opiration]) * 0.001)
            self.d_bias.append(np.sum(self.d_z[-1], axis=0, keepdims=True) * 0.001)

            self.d_a.append(np.dot(self.d_a[-1], self.wight[self.stack_func_opiration]))
            self.Derev_opiration += 1
        
        self.optin_time = False

    def ADAMoptim(self, Y, lerning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        time = 1

        if not hasattr(self, 'm_w'):
            self.m_w = [np.zeros_like(w) for w in self.wight]
            self.v_w = [np.zeros_like(w) for w in self.wight]
            self.m_b = [np.zeros_like(b) for b in self.bias]
            self.v_b = [np.zeros_like(b) for b in self.bias]

        self.optin_time = True
        self.Derev_opiration = 0
        d_a3 = self.binary_cros_entropy_drev(self.stack_opiration[-1], Y)
        self.d_a.append(d_a3)

        for layer, arg1, arg2 in reversed(self.layers):
            self.stack_func_opiration -= 1
            layer(arg1, arg2)

            self.d_wight.append(np.dot(self.d_z[-1].T, self.stack_opiration[self.stack_func_opiration]))
            self.d_bias.append(np.sum(self.d_z[-1], axis=0, keepdims=True))

            self.m_w[self.stack_func_opiration] = beta1 * self.m_w[self.stack_func_opiration] + (1-beta1) * self.d_wight[-1]
            self.v_w[self.stack_func_opiration] = beta2 * self.v_w[self.stack_func_opiration] + (1-beta2) * (self.d_wight[-1]**2)

            hat_m = self.m_w[self.stack_func_opiration] / (1-beta1 ** time)
            hat_v = self.v_w[self.stack_func_opiration] / (1-beta2 ** time)

            self.wight[self.stack_func_opiration] -= lerning_rate * hat_m / (np.sqrt(hat_v) + epsilon)

            self.m_b[self.stack_func_opiration] = beta1 * self.m_b[self.stack_func_opiration] + (1-beta1) * self.d_bias[-1]
            self.v_b[self.stack_func_opiration] = beta2 * self.v_b[self.stack_func_opiration] + (1-beta2) * (self.d_bias[-1]**2)

            bhat_m = self.m_b[self.stack_func_opiration] / (1-beta1 ** time)
            bhat_v = self.v_b[self.stack_func_opiration] / (1-beta2 ** time)

            self.bias[self.stack_func_opiration] -= lerning_rate * bhat_m / (np.sqrt(bhat_v) + epsilon)

            self.d_a.append(np.dot(self.d_a[-1], self.wight[self.stack_func_opiration]))
            
            
            self.Derev_opiration += 1
        
        time += 1
        self.optin_time = False

    def fit(self, epoch, batch_size):
        
        for i in range(epoch):

            for point in range(0, len(self.X_data[0]), batch_size):
                
                x_batch = self.X_data[0][point:point + batch_size]
                y_batch = self.Y_data[point: point + batch_size]

                for unit in range(len(x_batch[0])):

                    self.farword(x_batch[unit])
                    self.ADAMoptim(y_batch[unit])

                    self.d_a = []
                    self.d_bias = []
                    self.d_wight = []
                    self.d_z = []
    
    def show(self):
        prid = []

        for ind in range(len(x_point)):
            point = np.array([[x_point[ind]], [y_point[ind]]])
            self.farword(point)
            output = np.round(self.stack_opiration[-1])

            if output[0][0] == 1:
                prid.append('red')
            elif output[0][1] == 1:
                prid.append('blue')
            # elif output[0][2] == 1:
            #     prid.append('green')
            # elif output[0][3] == 1:
            #     prid.append('yellow')
            else:
                prid.append('black')
        plt.scatter(x_point, y_point, c=prid, marker="o", cmap='viridis')

        plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, marker='x', cmap='viridis')

        plt.show()


model = NN(data_set)

model.CreatLayers()
model.fit(1500, 20)
model.show()