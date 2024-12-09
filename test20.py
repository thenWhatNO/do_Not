import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

#################### the grid

x_min, x_max, y_min, y_max = -3.5, 3.5, -3.5, 3.5
step_size = 0.25

x_values = np.arange(x_min, x_max, step_size)
y_values = np.arange(y_min, y_max, step_size)
x_grid, y_grid = np.meshgrid(x_values, y_values)

x_point = x_grid.flatten()
y_point = y_grid.flatten()

#///////////// img data base

color = True

data_path = "plenet_data.csv"

df = pd.read_csv(data_path)
images= df['image']
labels = df['targ']

one_imag = Image.open('data' + images[0])

img_array_for_show = []
img_array = []

for i in images:
    img = Image.open('data'+ i)
    img_resize = img.resize((30,30))
    if color:
        convort = img_resize.convert('L')
    img_2_array = np.array(convort)
    if color:
        img_clear = np.where(img_2_array > 50.0, 1.0 ,100.0)
        cannal_up = img_clear[:, :, np.newaxis]
    img_one_shot = cannal_up.reshape(1, -1)
        # imf2float = np.zeros_like(cannal_up)
        # for i, img in enumerate(cannal_up):
        #     imf2float[i] = float(img)
    #img_array.append(img_one_shot[0])
    img_array_for_show.append(cannal_up.tolist())

def one_how(labels, num_class):
    return np.eye(num_class)[labels.astype(int)].tolist()

one_label = one_how(labels, 2)

data_set_photo = [img_array_for_show, one_label]


#///////////////////

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
        self.on_this = 0

        self.batch_num = 1
        self.filter = 1
        self.channal = 1

        self.Wight = []
        self.Bias = []

        self.Output = []
        self.Z_output = []

    def Layers(self):
        if self.optim_time:
            self.Dense(None, None, "sigmoid")
            self.Dense(None, None, "relu")
            self.Flatten()
            self.conv2d(3)
            self.conv2d(3)
            self.optim_time = False
            return
        self.conv2d(3)
        self.conv2d(3)
        wight_of_flatten = self.Flatten()
        self.Dense(wight_of_flatten, 20 , "relu")
        self.Dense(20 , 2, "sigmoid")

    def add_output_layer(self):
        self.Output = []
        self.Z_output = []

        for i in range(self.count_layers_num+1):
            self.Output.append([])
            self.Z_output.append([])
        self.Z_output.pop()

    def Creat(self):
        self.test_img = np.zeros((np.shape(self.X_data[0]))).tolist()

        self.count_layers_num = 0

        self.optim_time = False
        self.Creat_time = True
        self.Layers()
        self.Creat_time = False

    def Creat_param(self, neruals, weithg):
        np.random.seed(1)
        wight = np.random.randn(neruals, weithg) * np.sqrt(2. / weithg)
        bias = np.random.randn(1, neruals)

        self.Wight.append(wight.tolist())
        self.Bias.append(bias.tolist())

    def creat_kernel(self, size):
        kernel = np.random.randn(size, size, self.channal).tolist()

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

        return A.tolist()

    def Dense(self, input, output, activition_func):
        if self.Creat_time:
            self.count_layers_num += 1
            self.Creat_param(output, input)
            return
        
        if self.optim_time:
            Z_D = self.Output_drev[-1] * self.Activeit(self.Z_output[self.on_this], activition_func)
            self.W_D = np.dot(Z_D.T, self.Output[self.on_this]).tolist()
            self.B_D = np.sum(Z_D, axis=0, keepdims=True).tolist()

            self.Output_drev.append(np.dot(self.Output_drev[-1], self.Wight[self.on_this]).tolist())

            self.optim(self.optim_type)
            
            self.on_this -= 1
            return
        
        self.on_this += 1
        for i in self.Output[self.on_this-1]:
            Z = np.dot(i, np.array(self.Wight[self.on_this-1]).T) + self.Bias[self.on_this-1]
            list_Z = Z.tolist()
            self.Z_output[self.on_this-1].append(list_Z[0])

            A = self.Activeit(Z, activition_func)
            
            self.Output[self.on_this].append(A[0])

    def conv2d(self, kernel_size, activition_func, stride=1, padding=0):

        if self.Creat_time:
            self.count_layers_num += 1
            self.is_convFirst = True
            if not hasattr(self, "kernel"):
                self.kernel = []
            self.kernel_org_shape = 0
            self.creat_kernel(kernel_size)
            self.conv_optim = False

            self.Wight.append([0])
            self.Bias.append([0])

            input_image = self.test_img

            input_height, input_width, channals = np.shape(input_image)
            kernel_height, kernel_width, _= np.shape(self.kernel[self.on_this])

            output_height = (input_height - kernel_height) // stride + 1
            output_width = (input_width - kernel_width) // stride + 1

            
            output_test = np.zeros((output_width, output_height, channals)).tolist()
            self.test_img = output_test
            self.on_this += 1
            return
        
        input_image = np.array(self.Output[self.on_this])
        kernel_for_work = np.array(self.kernel[self.on_this])

        if self.optim_time:
            self.kernel_D.append([])
            kernel_height, kernel_width, channals_k= np.shape(kernel_for_work)
            batch_size, gradient_height, gradient_width, channals_c = np.shape(self.Output_drev[-1])

            work_OUT_dre = np.array(self.Output[self.on_this])
            A_D_out = np.array(self.Output_drev[-1])
            kernel_wpok = np.array(self.kernel[self.on_this])

            D_A = np.zeros_like(input_image)
            D_K = np.zeros_like(self.kernel[self.on_this])

            for z in range(0, batch_size):
                for y in range(0, gradient_height):
                    for x in range(0, gradient_width):
                        for o in range(0, channals_c):
                            h_start, w_start = y * stride, x * stride
                            h_end, w_end = h_start + kernel_height, w_start + kernel_height

                            region = work_OUT_dre[z,h_start:h_end,w_start:w_end, o]

                            D_K[:, :, o] += region * A_D_out[z,y,x,o]

                            D_A[z, h_start:h_end, w_start:w_end, o] += kernel_wpok[:, :, o] * A_D_out[z,y,x,o]



            self.kernel_D[-1] = D_K.tolist()
            teta = D_A.tolist()
            self.Output_drev.append(teta)
            
            self.conv_optim = True
            self.optim(self.optim_type)
            self.on_this -= 1
            self.conv_optim = False
            return
        
        #?///////////////////////////////////////

        kernel_height, kernel_width, cannals= np.shape(self.kernel[self.on_this])

        if padding > 0:
            input_image = np.pad(self.Output[self.on_this][-1], ((padding, padding), (padding, padding)), mode='constant').tolist()
    
    
        output = self.output_img_shapere(input_image, stride)
        batch_one, output_height, output_width, channels_img = np.shape(output)

        for z in range(0, batch_one):
            for y in range(0, output_height):
                for x in range(0, output_width):
                    for o in range(0, channels_img):
                        h_start, w_start = y * stride, x * stride
                        h_end, w_end = h_start + kernel_height, w_start + kernel_width
                        region = input_image[z, h_start:h_end, w_start:w_end, :]
                        output[z][y][x][o] = np.sum(region * kernel_for_work[:, :, o])

        self.on_this += 1
        A = self.Activeit(output, activition_func)
        self.Output[self.on_this] = A

    def output_img_shapere(self, input_image, stride):

        batch_size, input_height, input_width, cannals_num = np.shape(input_image)
        kernel_height, kernel_width, input_chanells= np.shape(self.kernel[self.on_this])

        output_height = (input_height - kernel_height) // stride + 1
        output_width = (input_width - kernel_width) // stride + 1

        shape_new = (batch_size, output_height, output_width, self.channal)

        output = np.zeros(shape_new).tolist()

        return output

    def Flatten(self):
        if self.Creat_time:
            self.count_layers_num += 1
            self.kernel_org_shape = np.shape(self.test_img)
            out_of = np.array(self.test_img).reshape(1, -1).tolist()

            self.Wight.append([0])
            self.Bias.append([0])

            return np.size(out_of)

        if self.optim_time:

            img_prr = self.Output_drev[-1]
            self.Output_drev.append([])
            for i in (img_prr if len(img_prr) > 1 else [img_prr]):
                ii = np.array(i)
                self.Output_drev[-1].append(ii.reshape(self.kernel_org_shape).tolist())

            self.on_this -= 1

            return
        
        self.on_this += 1

        for i in self.Output[self.on_this -1]:
            img = i

            flat = np.array(img).reshape(1, -1).tolist()
            self.Output[self.on_this].append(flat[0])

    def farword(self, X):
        if self.is_convFirst == False:
            X = X.reshape(-1, 1).T
        self.on_this = 0

        self.Output[self.on_this] = X
        self.Layers()
        
    def binary_cros_entropy_drev(self, y_prob, y_targ):
        return np.array(y_prob) - np.array(y_targ)

    def optim(self, optimaze_type, lerning_rate = 0.01):
        if optimaze_type == "SVM":
            self.SVM_optim_conv(lerning_rate) if self.conv_optim == True else self.SVM_optim(lerning_rate)
        if optimaze_type == "ADAM":
            self.Adam_optim_conv(lerning_rate) if self.conv_optim == True else self.ADAM_optim(lerning_rate)

    def SVM_optim(self, lerning_rate):
        self.Wight[self.on_this] -= self.W_D * lerning_rate
        self.Bias[self.on_this] -= self.B_D * lerning_rate

    def SVM_optim_conv(self, lerning_rate):
        self.kernel[self.on_this] -= self.kernel_D * lerning_rate
        
    def ADAM_optim(self, lerning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if not hasattr(self, "time"):
            self.time = 1

        workWD = np.array(self.W_D)
        workBD = np.array(self.B_D)

        m_w = [np.zeros_like(w) for w in self.Wight]
        v_w = [np.zeros_like(w) for w in self.Wight]
        m_b = [np.zeros_like(b) for b in self.Bias]
        v_b = [np.zeros_like(b) for b in self.Bias]

        m_w[self.on_this] = beta1 * m_w[self.on_this] + (1-beta1) * workWD[-1]
        v_w[self.on_this] = beta2 * v_w[self.on_this] + (1-beta2) * (workWD[-1]**2)

        hat_m = m_w[self.on_this] / (1-beta1 ** self.time)
        hat_v = v_w[self.on_this] / (1-beta2 ** self.time)

        self.Wight[self.on_this] -= lerning_rate * hat_m / (np.sqrt(hat_v) + epsilon).tolist()

        m_b[self.on_this] = beta1 * m_b[self.on_this] + (1-beta1) * workBD[-1]
        v_b[self.on_this] = beta2 * v_b[self.on_this] + (1-beta2) * (workBD[-1]**2)

        bhat_m = m_b[self.on_this] / (1-beta1 ** self.time)
        bhat_v = v_b[self.on_this] / (1-beta2 ** self.time)

        self.Bias[self.on_this] -= lerning_rate * bhat_m / (np.sqrt(bhat_v) + epsilon).tolist()
                
        self.time += 1

    def Adam_optim_conv(self, lerning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        working_K_D = np.array(self.kernel_D[-1])

        if not hasattr(self, "time"):
            self.time = 0

        if not hasattr(self, 'kenel'):
            kornel_M = [np.zeros_like(w) for w in self.kernel]
            kornel_V = [np.zeros_like(w) for w in self.kernel]

        kornel_M[self.on_this] = beta1 * kornel_M[self.on_this] + (1-beta1) * working_K_D
        kornel_V[self.on_this] = beta2 * kornel_V[self.on_this] + (1-beta2) * (working_K_D**2)

        hat_m = kornel_M[self.on_this] / (1-beta1 ** self.time)
        hat_v = kornel_V[self.on_this] / (1-beta2 ** self.time)

        self.kernel[self.on_this] -= lerning_rate * hat_m / (np.sqrt(hat_v) + epsilon).tolist()


    def fit(self, epoch, batch_size, optimizer_name):
        self.optim_type = optimizer_name
        self.batch_num = batch_size

        for i in range(epoch):
            for point in range(0, len(self.X_data), batch_size):
                self.add_output_layer()

                x_batch = self.X_data[point:point + batch_size]
                self.y_batch = self.Y_data[point:point + batch_size]

                self.batch_for_now = len(x_batch)
        
                self.farword(x_batch)

                self.batch_label = self.Output[-1]
 
                self.optim_time = True
                self.Output_drev = []
                A_drev = self.binary_cros_entropy_drev(self.batch_label, self.y_batch)
                self.Output_drev.append(A_drev)
                self.kernel_D = []

                self.on_this -= 1
                self.Layers()

        for i in range(5):
            R = np.random.randint(1, len(self.X_data))
            self.add_output_layer()
            test_x_data = np.array(self.X_data[R])
            imgaaaa = test_x_data[np.newaxis, :, :, :]
            test_y_data = self.Y_data[R]

            #print(test_x_data)

            self.farword(imgaaaa)
            print(np.round(self.Output[-1]))
            print(test_y_data)

model = NN(data_set_photo)
model.Creat()
model.fit(10, 5, 'ADAM')