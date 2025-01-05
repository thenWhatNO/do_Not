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

# data_path = "plenet_data.csv"

# df = pd.read_csv(data_path)
# images= df['image']
# labels = df['targ']

# one_imag = Image.open('data' + images[0])

# img_array_for_show = []
# img_array = []

# for i in images:
#     img = Image.open('data'+ i)
#     img_resize = img.resize((30,30))
#     if color:
#         convort = img_resize.convert('L')
#     img_2_array = np.array(convort)
#     if color:
#         img_clear = np.where(img_2_array > 50.0, 1.0 ,100.0)
#         cannal_up = img_clear[:, :, np.newaxis]
#     img_one_shot = cannal_up.reshape(1, -1)
#         # imf2float = np.zeros_like(cannal_up)
#         # for i, img in enumerate(cannal_up):
#         #     imf2float[i] = float(img)
#     #img_array.append(img_one_shot[0])
#     img_array_for_show.append(cannal_up.tolist())

# def one_how(labels, num_class):
#     return np.eye(num_class)[labels.astype(int)].tolist()

# one_label = one_how(labels, 2)

# data_set_photo = [img_array_for_show, one_label]

#//////////////////////////////////// ob dataset


data_path = "data_2/object_label.csv"

df = pd.read_csv(data_path)
images= df['image']
labels = df['targ']

one_imag = Image.open(images[0])

img_array_for_show = []
img_array = []

for i in images:
    print(i)
    img = Image.open(i)
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

one_label = one_how(labels, 3)

data_set_photo = [img_array_for_show, one_label]

#///////////////////////////////////


# data_path = "data_2/latin_label.csv"

# df = pd.read_csv(data_path)
# images= df['image']
# labels = df['targ']

# one_imag = Image.open('data_2/latin_data_jpg/' + images[0])

# img_array_for_show = []
# img_array = []

# for i in images:
#     print('data_2/latin_data_jpg/'+ i)
#     img = Image.open('data_2/latin_data_jpg/'+ i)
#     img_resize = img.resize((30,30))
#     if color:
#         convort = img_resize.convert('L')
#     img_2_array = np.array(convort)
#     if color:
#         img_clear = np.where(img_2_array > 50.0, 1.0 ,100.0)
#         cannal_up = img_clear[:, :, np.newaxis]
#     img_one_shot = cannal_up.reshape(1, -1)
#         # imf2float = np.zeros_like(cannal_up)
#         # for i, img in enumerate(cannal_up):
#         #     imf2float[i] = float(img)
#     #img_array.append(img_one_shot[0])
#     img_array_for_show.append(cannal_up.tolist())

# def one_how(labels, num_class):
#     return np.eye(num_class)[labels.astype(int)].tolist()

# one_label = one_how(labels, 26)

# data_latine_digets = [img_array_for_show, one_label]


#///////////////////

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

# #plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, cmap='viridis')

# data_set1 = [data_x, data_y_one_hot]


###################### the other function #########################

def softmax(logits):
    logits_exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)

def softmax_derivative(softmax_output):
    out = np.array([1])
    return out

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
        self.Work_x, self.Work_Y = [], []
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
            self.Dense(None, None, "softmax")
            self.Dense(None, None, "relu")
            self.Dense(None, None, "relu")
            self.Flatten()
            self.conv2d(3, "relu")
            self.poolingMax()
            self.conv2d(3, "relu")
            self.optim_time = False
            return
        self.conv2d(3, "relu")
        self.poolingMax()
        self.conv2d(3, "relu")
        wight_of_flatten = self.Flatten()
        self.Dense(wight_of_flatten, 40 , "relu")
        self.Dense(40, 20, "relu")
        self.Dense(20 , 3, "softmax")

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
        if activation == "softmax":
            A = softmax_derivative(Z) if self.optim_time else softmax(Z)

        return A.tolist()

    def Dense(self, input, output, activition_func):
        if self.Creat_time:
            self.count_layers_num += 1
            self.Creat_param(output, input)
            return
        
        if self.optim_time:
            Z_D = np.array(self.Output_drev[-1]) * self.Activeit(np.array(self.Z_output[self.on_this]), activition_func)
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

            activ_drev = np.array(self.Activeit(self.Z_output[self.on_this+1], activition_func))

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

                            D_K[:, :, o] += region * A_D_out[z,y,x,o] * activ_drev[z,y,x,o]

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
        self.Z_output[self.on_this] = output
        A = self.Activeit(np.array(output), activition_func)
        self.Output[self.on_this] = A

    def poolingMax(self, steps=2):

        if self.Creat_time:
            self.count_layers_num += 1

            self.Wight.append([0])
            self.Bias.append([0])
            self.kernel.append([0])

            test_img = self.test_img
            input_height, input_width, channals = np.shape(test_img)

            test_output_height = (input_height - steps) // steps + 1
            test_output_width = (input_width - steps) // steps + 1

            out_test = np.zeros((test_output_height, test_output_width, channals)).tolist()
            self.test_img = out_test

            self.on_this += 1

            return

        if self.optim_time:
            self.poolimgMax_drev()
            self.on_this -= 1
        
            return

        working_img = np.array(self.Output[self.on_this])

        batch, input_height, input_width, cannal = working_img.shape

        output_height = (input_height - steps) // steps + 1
        output_width = (input_width - steps) // steps + 1

        output_image = np.zeros((batch, output_height, output_width, cannal))

        for b in range(0, batch):
            for y in range(0, output_height):
                for x in range(0, output_width):
                    for c in range(0, cannal):
                        region = working_img[b, y*steps:y*steps+steps, x*steps:x*steps+steps, c]
                        opa = np.max(region)
                        output_image[b, y, x, c] = opa

        self.on_this += 1
        
        teta = output_image.tolist()
        self.Output[self.on_this] = teta

    def poolimgMax_drev(self, steps=2):

        working_img = np.array(self.Output_drev[-1])

        batch, input_height, input_width, cannal = working_img.shape

        find = 1

        useg_height = (input_height - steps) // steps + 1
        useg_width = (input_width - steps) // steps + 1

        output_image = np.zeros((batch, input_height, input_width, cannal))

        for b in range(0, batch):
            for y in range(0, useg_height):
                for x in range(0, useg_width):
                    for c in range(0, cannal):
                        region = working_img[b, y*steps:y*steps+steps, x*steps:x*steps+steps, c]
                        maxy = np.max(region)
                        wer = np.where(region == maxy)
                        wer_list = list(zip(wer[0], wer[1]))
                        output_image[b, y*steps+wer_list[0][0], x*steps+wer_list[0][1], c] = find
                        find += 1
        
        teta = output_image.tolist()
        self.Output_drev.append(teta)

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

    def shuffel(self):
        assert len(self.Y_data) == len(self.X_data)
        indexs = np.arange(len(self.X_data))
        np.random.shuffle(indexs)

        self.Work_Y = np.array(self.Y_data)[indexs].tolist()
        self.Work_x = np.array(self.X_data)[indexs].tolist()

    def split(self, x_data, y_data):
        split_num = 0.8
        split_use = int(len(x_data) * split_num)

        val_data_spli = x_data[split_use:]
        val_data_split_y = y_data[split_use:]
        train_data_split = x_data[:split_use]
        train_data_split_y = y_data[:split_use]

        return val_data_spli, val_data_split_y, train_data_split, train_data_split_y
        
    def binary_cros_entropy_drev(self, y_prob, y_targ):
        return (np.array(y_prob) - np.array(y_targ)).tolist()

    def other_binary_cros_entropy_drev(self, y_prob, y_targ):
        return ((np.array(y_prob) - np.array(y_targ))**2).tolist()
    
    def drev_other_binary_cros_entropy_drev(self, y_prob, y_targ):
        return (2*np.array(y_targ)).tolist()
    
    def categorical_cross_entropy(self, y_true, y_targ):

        epsilon = 1e-15
        y_targ = np.clip(y_targ, epsilon, 1 - epsilon)  # Clip predictions
        loss = -np.sum(y_true * np.log(y_targ)) / np.array(y_true).shape[0]
        return loss
    
    def categorical_cross_entropy_derivative(self, y_true, y_pred):
        return (np.array(y_pred) - np.array(y_true)).tolist()
    
#/////////////// end of the loss function

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

    def show_model_prog(self, loss, scatter, epoch, line, ax):
        sublist_means = [np.mean(sublist) for sublist in loss]

        if self.val_in_chat:
            sublist_means = np.mean(loss)
            self.color.append(0.5)
        else:
            self.num += 1
            self.color.append(0)

        self.all_loss[1].append(np.mean(sublist_means))
        self.all_loss[0].append(self.op)
        self.op += 1

        print("val test -- " if self.val_in_chat else self.num, "epoch past from :", epoch , "loss : ", self.all_loss[1][-1])

        line.set_xdata(self.all_loss[0])
        line.set_ydata(self.all_loss[1])

        scatter.set_offsets(list(zip(self.all_loss[0], self.all_loss[1])))
        scatter.set_array(self.color)

        ax.set_xlim(0, max(self.all_loss[0]) + 1)
        ax.set_ylim(min(self.all_loss[1]) - 1, max(self.all_loss[1]) + 1)

        plt.draw()
        plt.pause(0.01)
    
    def fit(self, epoch, batch_size, optimizer_name):
        self.optim_type = optimizer_name
        self.batch_num = batch_size

        self.val_in_chat = False

        fig, ax = plt.subplots()
        self.color = []
        line, = ax.plot([], [], linestyle='-', color='gray', label="Loss progresion")  # Initial empty plot
        scatter = ax.scatter([], [], c=[], cmap='viridis')
        ax.set_xlim(0, 10)  # Fixed x-axis range
        ax.set_ylim(0, 10)  # Fixed y-axis range
        ax.legend()

        self.all_loss = [[],[]]
        self.op = 0
        self.num = 0

        self.shuffel()
        self.val_x, self.val_y, self.train_x, self.train_y = self.split(self.Work_x, self.Work_Y)

        for i in range(epoch):
            loss = []
            for point in range(0, len(self.train_x), batch_size):

                self.add_output_layer()

                x_batch = self.train_x[point:point + batch_size]
                self.y_batch = self.train_y[point:point + batch_size]

                self.batch_for_now = len(x_batch)
        
                self.farword(x_batch)

                self.batch_label = self.Output[-1]

                loss.append(self.categorical_cross_entropy(self.y_batch, self.batch_label))
 
                self.optim_time = True
                self.Output_drev = []
                A_drev = self.categorical_cross_entropy_derivative(self.y_batch ,self.batch_label)
                self.Output_drev.append(A_drev)
                self.kernel_D = []

                self.on_this -= 1
                self.Layers() 

#/////////////////////////////////////////////////////matplot after the main train epoch
            self.show_model_prog(loss, scatter, epoch, line, ax)

            self.add_output_layer()
            val_test = []
            val_test_y = []
            self.val_in_chat = True
            for num in range(0, batch_size):
                R = np.random.randint(0, len(self.val_x))
                
                val_test.append(self.val_x[R])
                val_test_y.append(self.val_y[R])

            self.farword(val_test)
            loss.append(self.categorical_cross_entropy(val_test_y, self.Output[-1]))

            self.show_model_prog(loss, scatter, epoch, line, ax)
            self.val_in_chat = False

        
        plt.ioff()
        plt.show()

        for i in range(10):
            R = np.random.randint(1, len(self.X_data))
            self.add_output_layer()
            test_x_data = np.array(self.X_data[R])
            imgaaaa = test_x_data[np.newaxis, :, :, :]
            test_y_data = self.Y_data[R]

            #print(test_x_data)

            self.farword(imgaaaa)
            print(self.Output[-1][0])
            print(test_y_data)

            plt.imshow(test_x_data)
            plt.show()

model = NN(data_set_photo)
model.Creat()
model.fit(10, 7, 'ADAM')