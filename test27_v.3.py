import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import sys

toke_lib = "tokens.csv"

def one_how(labels, num_class):
    return np.eye(num_class)[labels.astype(int)]

##################### ---------- Data meneger class ------------



class Data_meneger:
    def __init__(self, data_X, data_Y, train=False, split_procent=0.8):
        self.train = train

        self.data_X = data_X
        self.data_Y = data_Y

        self.start_data_X = data_X
        self.start_data_Y = data_Y

        self.train_data_X = None
        self.train_data_y = None

        self.validation_data_X = None
        self.validation_data_y = None

        self.test_data_X = None
        self.test_data_y = None

        self.split_train_val(split_procent=split_procent)

    def restart_data(self):
        self.data_X = self.start_data_X
        self.data_Y = self.start_data_Y
    
    def Shuffel(self):
        assert len(self.train_data_X) == len(self.train_data_y)
        train_index = np.arange(len(self.train_data_X))
        np.random.shuffle(train_index)
        self.train_data_X = np.array(self.train_data_X)[train_index]
        self.train_data_y = np.array(self.train_data_y)[train_index]

        assert len(self.validation_data_X) == len(self.validation_data_y)
        validation_index = np.arange(len(self.validation_data_X))
        np.random.shuffle(validation_index)
        self.validation_data_X = np.array(self.validation_data_X)[validation_index]
        self.validation_data_y = np.array(self.validation_data_y)[validation_index]

        if self.train:
            assert len(self.test_data_X) == len(self.test_data_y)
            test_index = np.arange(len(self.test_data_X))
            np.random.shuffle(test_index) 
            self.test_data_X = np.array(self.test_data_X)[test_index]
            self.test_data_y = np.array(self.test_data_y)[test_index]
    
    def split_train_val(self, split_procent):
        split_use = int(len(self.data_X) * split_procent)
        if self.train:
            split_val_test= 2 // split_use

            self.test_data_X = self.data_X[:split_val_test]
            self.test_data_y = self.data_Y[:split_val_test]
            self.validation_data_X = self.data_X[split_val_test:split_use]
            self.validation_data_y = self.data_Y[split_val_test:split_use]
            self.train_data_X = self.data_X[:split_use]
            self.train_data_y = self.data_Y[:split_use]

        else:
            self.validation_data_X = self.data_X[split_use:]
            self.validation_data_y = self.data_Y[split_use:]
            self.train_data_X = self.data_X[:split_use]
            self.train_data_y = self.data_Y[:split_use]
    
    def get_train_data(self):
        return [self.train_data_X, self.train_data_y]
    
    def get_validation_data(self):
        return [self.validation_data_X, self.validation_data_y]
    
    def get_test_data(self):
        return [self.test_data_X, self.test_data_y]

    def get_train_batch(self, batch_size):
        batechs_list = []
        
        for batch in range(0, len(self.train_data_X), batch_size):
            data_batch_X = self.train_data_X[batch:batch+batch_size]
            data_batch_Y = self.train_data_y[batch:batch+batch_size]

            batechs_list.append([data_batch_X, data_batch_Y])
        return batechs_list

    def get_validation_batch(self, batch_size):
            batechs_list = []
            
            for batch in range(0, len(self.validation_data_X), batch_size):
                data_batch_X = self.validation_data_X[batch:batch+batch_size]
                data_batch_Y = self.validation_data_y[batch:batch+batch_size]

                batechs_list.append([data_batch_X, data_batch_Y])
            return batechs_list

    def get_test_batch(self, batch_size):
            batechs_list = []
            
            for batch in range(0, len(self.test_data_X), batch_size):
                data_batch_X = self.test_data_X[batch:batch+batch_size]
                data_batch_Y = self.test_data_y[batch:batch+batch_size]

                batechs_list.append([data_batch_X, data_batch_Y])
            return batechs_list











##################### ---------- clasture data ------------


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

data_y_one_hot = one_how(data_y, n_clusters)

# plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, cmap='viridis')
# plt.show()

clastur_data = Data_meneger(data_x, data_y_one_hot)



##################### ---------- token/text data ------------

data_path = "data_2/words_label.csv"

df = pd.read_csv(data_path)
sentens= df['sentenc'].tolist()
labels = np.array(df['targ'])

eye_labols = one_how(labels, 2)

word_data = Data_meneger(sentens, eye_labols)


##################### ---------- img data object detection ------------
color = True
data_path = "data_2/object_label.csv"

df = pd.read_csv(data_path)
images= df['image']
labels = df['targ']

one_imag = Image.open(images[0])

img_array_for_show = []
img_array = []

for i in images:
    img = Image.open(i)
    img_resize = img.resize((30,30))
    if color:
        convort = img_resize.convert('L')
    img_2_array = np.array(convort)
    if color:
        img_clear = np.where(img_2_array > 50.0, 1.0 ,100.0)
        cannal_up = img_clear[:, :, None]
    img_one_shot = cannal_up.reshape(1, -1)
        # imf2float = np.zeros_like(cannal_up)
        # for i, img in enumerate(cannal_up):
        #     imf2float[i] = float(img)
    #img_array.append(img_one_shot[0])
    img_array_for_show.append(cannal_up.tolist())

one_label = one_how(labels, 3)

data_set_photo_num_ob = Data_meneger(img_array_for_show, one_label)


###////////////////---activation function---////////////////////

class Relu:
    def __init__(self):
        pass
    def run(self, x):
        return np.where(x >= 0, x, 0.001*x)
    
    def derivative(self, x):
        return np.where(x >= 0, 1, 0.001)
    

class Sigmoid:
    def __init__(self):
        pass
    def run(self, x):
        x = np.array(x)
        if np.any(x > 300) or np.any(x < -300):
            x = np.clip(x, -300, 300) + np.random.normal(0, 0.1, x.shape)
        return  1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        sig = self.run(x)
        return sig * (1 - sig)

class swish:
    def __init__(self):
        pass
    def run(self, x, beta = 1):
        return x / (1 + np.exp(-beta * x))
    
    def derivative(self, x, beta = 1):
        x = np.array(x)
        sig = 1 / (1 + np.exp(-beta * x))
        return beta * sig * (1 - sig) + sig
    
    
class softmax:
    def __init__(self):
        pass
    def run(self, x):
        logits_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)
    
    def derivative(self, x): # acually the full function is the loos function that used in the momdel.
        return np.array([1])
    

class mish:
    def __init__(self):
        pass
    def run(self, x):
        return x * np.tanh(np.log1p(np.exp(x)))
    
    def derivative(self, x):
        x = np.array(x)
        omega = 4 * (x - 1) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x)
        delta = 2 * np.exp(x) + np.exp(2 * x) + 2
        return np.exp(x) * omega / (delta ** 2 )



###////////////////---loss function---////////////////////



class BinaryCrossEntropy:
    def  __init__(self):
        pass
    def loss(self, y_prob, y_targ):
        epsilon = 1e-9 
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        return -np.mean(y_targ * np.log(y_prob) + (1 - y_targ) * np.log(1 - y_prob))

    def derivative(self, y_prob, y_targ):
        epsilon = 1e-9
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        return (y_prob - y_targ) / (y_prob * (1 - y_prob))

    def second_derivative(self, y_prob, y_targ):
        epsilon = 1e-9
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        return (1 - 2 * y_targ) / (y_prob**2 * (1 - y_prob)**2)


class categorical_cross_entropy:
    def  __init__(self):
        pass
    def loss(self, y_true, y_targ):
        epsilon = 1e-15
        y_targ = np.clip(y_targ, epsilon, 1 - epsilon)  # Clip predictions
        return -np.sum(y_true * np.log(y_targ)) / np.array(y_true).shape[0]
    
    def derivative(self, y_true, y_targ):
        return (np.array(y_targ) - np.array(y_true))




###////////////////---leyars---////////////////////


class Dense:
    def __init__(self, input, output, activation_func=None, flatten_befor=False):
        self.output = output
        self.Wight = np.random.randn(output, input) * np.sqrt(2. / input)
        self.Bios = np.random.randn(1, output)
        self.activation_func = activation_func
        self.flatten_befor = flatten_befor
        self.Z_out = None
        self.A_out = None
        self.X = None

    def run(self, X):
        self.X = X
        if self.flatten_befor:
            self.Wight = np.random.randn(self.output, np.shape(X)[-1]) * np.sqrt(2. / np.shape(X)[-1])
            self.flatten_befor = False

        self.Z_out = np.matmul(X, self.Wight.T) + self.Bios

        self.mean = np.mean(self.Z_out, keepdims=True)
        self.std = np.std(self.Z_out, axis=0, keepdims=True) + 1e-7  # Avoid division by zero
        Z_out = (self.Z_out - self.mean) / self.std

        if self.activation_func != None:
            self.A_out = self.activation_func.run(Z_out)
            return self.A_out
        return self.Z_out
    
    def optim(self, gradint):
        D_Z = gradint
        if self.activation_func != None:
            D_Z = gradint * self.activation_func.derivative(self.Z_out)
        D_W = np.matmul(D_Z.T, self.X)
        D_B = np.sum(D_Z, axis=0, keepdims=True)
        D_A = np.matmul(gradint, self.Wight)

        return [D_W, D_B], D_A
    
    def update_param(self, parameters):
        self.Wight -= parameters[0]
        self.Bios -= parameters[1]


class Reshape_output:
    def __init__(self, shape=()):
        self.X = None
        self.X_shape = None
        self.new_shape = shape

    def run(self, X):
        self.X_shape = X.shape
        try:
            return np.reshape(X, self.new_shape)
        except Exception as e:
            raise

    def optim(self, gradint):
        gradint = np.reshape(gradint, (self.X_shape))
        return [None], gradint
    
    def update_param(self, parameters):
        pass


class Grid:
    def __init__(self, grid_size=[], flatten_befor=False):
        self.flatten_befor = flatten_befor
        self.X = None
        self.grid_size = grid_size

    def run(self, X):
        self.X = X

        self.is_grid = True
        boxes = []

        batch_size, input_height, input_width, cannals_num =  np.shape(X)

        H_jump = int(input_height / self.grid_size[0])
        V_jump = int(input_width / self.grid_size[1])

        boxes = np.zeros((batch_size, self.grid_size[0] * self.grid_size[1], H_jump, V_jump, cannals_num))

        for b in range(0, batch_size):
            box=0
            for h in range(0, input_height, H_jump):
                for v in range(0, input_width, V_jump):
                    h_end, v_end = h + H_jump, v + V_jump

                    test_1 = boxes[b, box, :, :, :].tolist()
                    test = X[b, h:h_end, v:v_end, :].tolist()

                    boxes[b, box, :, :, :] += X[b, h:h_end, v:v_end, :]
                    box+=1
        return boxes
    
    def optim(self, gradint):
        X = gradint
        Batch, G, Y, x, C = X.shape
        assert G == self.grid_size[0] * self.grid_size[1], "ahhhhhhh"

        X = X.reshape(Batch, self.grid_size[0], self.grid_size[1], Y, x, C)
        X = X.transpose(0,1,3,2,4,5)
        X = X.reshape(Batch, self.grid_size[0]*Y, self.grid_size[1]*x, C)
        return [None], X
    
    def update_param(self, parameters):
        pass


class Conv2D:
    def __init__(self, kernel_size, activation_func, grid=False, filter_num = 1, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel = np.random.randn(filter_num ,kernel_size[1], kernel_size[0])
        self.Z_out = None
        self.A_out = None
        self.activation_func = activation_func
        self.input = None
        self.grid = grid
    
    def region(self, X, b, x, y):
        y_start, x_start = y*self.stride, x*self.stride
        y_end, x_end = y_start + np.shape(self.kernel)[1], x_start + np.shape(self.kernel)[2]

        if self.grid:
            return X[b, :, y_start:y_end, x_start:x_end, :]
        if not self.grid:
            return X[b, y_start:y_end, x_start:x_end, :]


    def run(self, X):
        if self.padding > 0:
            X = np.pad(X, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        self.input = X # chacke if the data is gridet
        if len(X.shape) > 4:
            self.grid = True

        shape_input = X.shape

        O_H = (shape_input[1] - np.shape(self.kernel)[1]) // self.stride + 1
        O_W = (shape_input[2] - np.shape(self.kernel)[2]) // self.stride + 1
        
        output = np.zeros((shape_input[0], O_H, O_W, np.shape(self.kernel)[0]))
        if self.grid:
            O_H = (shape_input[2] - np.shape(self.kernel)[1]) // self.stride + 1
            O_W = (shape_input[3] - np.shape(self.kernel)[2]) // self.stride + 1
            output = np.zeros((shape_input[0], shape_input[1], O_H, O_W, np.shape(self.kernel)[0]))
        
        for b in range(shape_input[0]):
            for y in range(0, O_W):
                for x in range(0, O_H):
                    for c in range(0, np.shape(self.kernel)[0]):
                        region = self.region(X, b, x, y)
                        if self.grid:
                            for box in range(shape_input[1]):
                                output[b, box, y, x, c] += np.sum(region[box] * self.kernel[c])
                        else:
                            output[b, y, x, c] += np.sum(region * self.kernel[c])

        self.Z_out = output
        self.A_out = self.activation_func.run(output)
        return self.A_out

    def optim(self, gradint):
        D_A = np.zeros(self.input.shape)
        D_K = np.zeros(self.kernel.shape)
        gradint = np.array(gradint)
        D_Z = self.activation_func.derivative(self.Z_out)
        X = self.input
        
        for b in range(0, np.shape(gradint)[0]):
            for y in range(0, np.shape(gradint)[-3]):
                for x in range(0, np.shape(gradint)[-2]):
                    for c in range(0, np.shape(self.kernel)[0]):
                        region = self.region(X, b, y, x)

                        if self.grid:
                            teta = (gradint[b,:,y,x] * D_Z[b,:,y,x])
                            opa = self.kernel[c,None,:,:] * teta[:,:,None]
                            D_K[c] += np.sum(region * teta[:, :, None, None])
                            D_A[b, :,  y:y+self.kernel.shape[0],   x:x+self.kernel.shape[1]] += opa[:,:,:,None]
                        else:
                            teta = (gradint[b,y,x] * D_Z[b,y,x])
                            opa = self.kernel[c,:,:] * teta[:,None]
                            D_K[c] += np.sum(region * teta[:, None, None])
                            testy = D_A[b, y:y+self.kernel.shape[0],   x:x+self.kernel.shape[1]]
                            D_A[b, y:y+self.kernel.shape[1],   x:x+self.kernel.shape[2]] += opa[:,:,None]

        return [D_K], D_A

    def update_param(self, parameters):
        self.kernel -= parameters[0]


class poolingMax:
    def __init__(self, steps=2):
        self.X = None
        self.steps = steps
        self.X_shape = None
        self.grid = False
    
    def region(self, X, w, h, batch, channal):
        start_w, start_h = h*self.steps, w*self.steps
        end_w, end_h = h*self.steps+self.steps, w*self.steps+self.steps

        if self.grid:
            return X[batch, :, start_h:end_h, start_w:end_w, channal]
        if not self.grid:
            return X[batch, start_h:end_h, start_w:end_w, channal]

    def run(self, X):
        self.X = X
        self.X_shape = np.shape(X)
        if len(self.X_shape) > 4:
            self.grid = True

        shape_num = self.X_shape

        O_H = (shape_num[2] - self.steps) // self.steps + 1
        O_W = (shape_num[1] - self.steps) // self.steps + 1
        
        output = np.zeros((shape_num[0], O_H, O_W, shape_num[-1]))
        if self.grid:
            output = np.zeros((shape_num[0], shape_num[1], O_H, O_W, shape_num[-1]))

        for img in range(shape_num[0]):
                for h in range(0, np.shape(output)[-3]):
                    for w in range(0, np.shape(output)[-2]):
                        for c in range(0, np.shape(output)[-1]):
                            region = self.region(X, w, h, img, c)
                            if self.grid:
                                for box in range(0, np.shape(output)[1]):
                                    max_find = np.max(region[box])
                                    output[img, box, h, w, c] = max_find
                            else:
                                max_found = np.max(region)
                                output[img, h, w, c] = max_found
        self.Z = output

        return output

    def optim(self, gradint):

        output = np.zeros_like(self.X)

        for b in range(0, np.shape(output)[0]):
                for y in range(0, np.shape(gradint)[-3]):
                    for x in range(0, np.shape(gradint)[-2]):
                        for c in range(0, np.shape(output)[-1]):
                            region = self.region(self.X, y, x, b, c)

                            if self.grid:
                                for i, img in enumerate(region):
                                    found_max = np.where(img == np.max(img))
                                    found_list = list(zip(found_max[0], found_max[1]))
                                    output[b, i, y*self.steps+found_list[0][0],  x*self.steps+found_list[0][1],    c] = gradint[b, i, y, x, c]
                            else:
                                found_max = np.where(region == np.max(region))
                                found_list = list(zip(found_max[0], found_max[1]))
                                output[b, y*self.steps+found_list[0][0],  x*self.steps+found_list[0][1],    c] = gradint[b, y, x, c]

        return [None], output
    
    def update_param(self, parameters):
        pass


class Flatten:
    def __init__(self):
        self.X = None
        self.X_shape = None

    def run(self, X):
        self.X = X
        self.X_shape = np.shape(X[0])

        new_output = []
        for b in X:
            flat = np.array(b).reshape(1,-1)
            new_output.append(flat[0])
        return np.array(new_output) 
    
    def optim(self, gradint):
        output_drev = []
        for i in gradint:
            i = np.reshape(i, (self.X_shape))
            output_drev.append(i)
        return [None], output_drev

    
    def update_param(self, parameters):
        pass


class multi_head_attention:
    def __init__(self, head_num, non_masked=False, add=False):
        self.Q_w = np.random.randn(4, 4) * np.sqrt(2. / 4)
        self.K_w = np.random.randn(4, 4) * np.sqrt(2. / 4)
        self.V_w = np.random.randn(4, 4) * np.sqrt(2. / 4)
        self.O_w = np.random.randn(4, 4) * np.sqrt(2. / 4)
        self.X = None
        self.output = None
        self.head_num = head_num
        self.add = add

        self.masked = non_masked
        self.encoder_X = None

    def split_or_mix(self, X, num_heads, action):
        if action == 'split':
            batch_size, word_num, token_num = X.shape
            depth_per_head = token_num // num_heads
            X = X.reshape(batch_size, word_num, num_heads, depth_per_head)

            return np.transpose(X, axes=(0, 2, 1, 3))
        
        if action == 'mix':

            batch_size, num_heads, word_num, depth_per_head = X.shape
            d_model = num_heads * depth_per_head
            X = np.transpose(X, axes=(0, 2, 1, 3))

            return X.reshape(batch_size, word_num, d_model)

    def run(self, X, enencoder_X=0):
        self.X = X
        self.encoder_X = enencoder_X
        self.D_model = np.shape(X)[-1]
        depth_per_head = self.D_model // self.head_num

        if self.masked:
            K = np.matmul(self.encoder_X, self.K_w)
            V = np.matmul(self.encoder_X, self.V_w)
        else:
            K = np.matmul(X, self.K_w)
            V = np.matmul(X, self.V_w)
        Q = np.matmul(X, self.Q_w)

        self.Q_heads = self.split_or_mix(Q, self.head_num, "split")
        self.K_heads = self.split_or_mix(K, self.head_num, "split")
        self.V_heads = self.split_or_mix(V, self.head_num, "split")

        output = []

        self.score = []
        self.attantion_w = []
        self.attantion_out = []

        scaling = np.sqrt(depth_per_head)
        for i in range(self.head_num):
            self.score.append(np.matmul(self.Q_heads[:, i], self.K_heads[:, i].transpose(0,2,1)) / scaling)
            self.attantion_w.append(softmax.run(None, self.score[i]))
            self.attantion_out.append(np.matmul(self.attantion_w[i], self.V_heads[:, i]))

        self.attantion_out = np.stack(self.attantion_out, axis=1)
        self.combain_out = self.split_or_mix(self.attantion_out, self.head_num, "mix")

        O = np.dot(self.combain_out, self.O_w)

        if self.add:
            O = X + O

        return O
    
    def optim(self, gradint):
        D_O = np.matmul(gradint, self.O_w.swapaxes(-1,-2))
        D_wo = np.matmul(self.combain_out.reshape(-1, self.D_model).T, D_O.reshape(-1, self.D_model))
        split_D_O = self.split_or_mix(D_O, self.head_num, "split")
        d_k = np.zeros_like(self.K_heads)
        d_q = np.zeros_like(self.Q_heads)
        d_v = np.zeros_like(self.V_heads)

        for i in range(self.head_num):
            d_v[:,i] = np.matmul(self.attantion_w[i], split_D_O[:,i,:,:])

            d_attantion_w = np.matmul(split_D_O[:,i,:,:], self.V_heads[:,i].swapaxes(-1,-2))
            d_scale = d_attantion_w * softmax.derivative(None, self.attantion_w[i])

            d_q[:,i] = np.matmul(d_scale, self.K_heads[:,i])
            d_k[:,i] = np.matmul(d_scale, self.Q_heads[:,i])

        d_k = self.split_or_mix(np.array(d_k), self.head_num, "mix")
        d_q = self.split_or_mix(np.array(d_q), self.head_num, "mix")
        d_v = self.split_or_mix(np.array(d_v), self.head_num, "mix")

        d_k = np.sum(d_k, axis=0)
        d_q = np.sum(d_q, axis=0)
        d_v = np.sum(d_v, axis=0)

        d_A = np.matmul(d_q, self.Q_w.T) + np.matmul(d_k, self.K_w.T) + np.matmul(d_v, self.V_w)

        if self.masked:
            return [d_q, d_k, d_v, D_wo], d_A, (d_k @ self.K_w.T) + (d_v @ self.V_w.T)
        return [d_q, d_k, d_v, D_wo], d_A

    def update_param(self, parameters):
        self.Q_w -= parameters[0]
        self.K_w -= parameters[1]
        self.V_w -= parameters[2]
        self.O_w -= parameters[3]


class positional_encoding:
    
    def __init__(self):
        pass

    def run(self, X):
        word_n, token_n = np.shape(X)[-2], np.shape(X)[-1]
        output = np.zeros((word_n, token_n))

        for pos in range(word_n):
            for i in range(0, token_n, 2):
                output[pos, i] = np.sin(pos / (10000 ** (i / token_n)))
                if i + 1 < token_n:
                    output[pos, i+1] = np.cos(pos / (10000 ** (i / token_n)))
        return (X + output)
    
    def optim(self, gradint):
        return [None], gradint
    
    def update_param(self, parameters):
        pass



class Embedding:
    def __init__(self, link, updata_token=False):
        self.link = link
        self.tabel = pd.read_csv(link)
        self.X = None
        self.updata_token = updata_token

    def add_new_words(self, words):
        for word in words:
            if not (self.tabel["word"] == word).any():
                token = np.random.randn(4)
                token = np.array2string(token, separator=',')
                self.tabel.loc[len(self.tabel)] = [word, self.tabel['id'].iloc[-1]+1, token]

        self.tabel.to_csv('tokens.csv', index=False)

        print("the program stap work becouse of new data get added to the tokkins data \nreset the program to keep work")
        sys.exit()

    def run(self, X):
        self.X = X
        self.output = []

        for sentens in X:
            input_word = sentens.strip().split()
            row = self.tabel[self.tabel['word'].isin(input_word)]
            word = row["word"].tolist()
            tabel_words = self.tabel["word"].values.tolist()

            if word in tabel_words:
                self.add_new_words(input_word)
            
            yesy = row["token"].tolist()

            token = []
            for wod in input_word:
                element = row[row['word'] == wod]
                token.append(element["token"].tolist()[0])
            token_correct = np.array([np.genfromtxt([i.strip("[]")], delimiter=",") for i in token])
            if token_correct.size == 0:
                token_correct = np.zeros([1,4])
            self.output.append(token_correct)
        return self.output
    
    def optim(self, gradint):
        output = []

        self.output -= np.array(gradint) * 0.01


        if self.updata_token:
            for i, sentens in enumerate(self.X):
                input_word = sentens.strip().split()
                row = self.tabel[self.tabel['word'].isin(input_word)]

                id = []
                token_data = self.output[i]
                for wordy in input_word: ## work to make id like word in teh run function
                    el = row[row['word'] == wordy]
                    id.append(el["id"].tolist()[0])

                for i, word in enumerate(input_word):
                    match_row = self.tabel.loc[self.tabel['word'].isin([word]), "token"]
                    if not match_row.empty:
                        update = np.array2string(token_data[i], separator=',')
                        self.tabel.at[id[i], "token"] = update
            self.tabel.to_csv(self.link, index=False)
        return [None], output
    
    def update_param(self, parameters):
        pass


class normalization:
    def __init__(self, out=None, epsilon=1e-6):
        self.W1 = []
        self.W2 = []
        self.epsilon = epsilon
        self.X = None
        self.out = out

    def creat_param(self, inp, outp):
        self.W1 = np.random.randn(outp, inp) * np.sqrt(2. / 4)
        self.W2 = np.random.randn(1, inp) * np.sqrt(2. / 4)

    def run(self, X):
        self.X = X
        mean_X = np.mean(X)
        self.std = np.std(X)
        self.X_normal = (X - mean_X) / (self.std + self.epsilon)

        if np.size(self.W1) == 0 or np.size(self.W2) == 0:
            self.creat_param(X.shape[-1], X.shape[-1]) if self.out == None else self.creat_param(X.shape[-1], self.out)

        output = np.matmul(self.X_normal, self.W1) + self.W2
        return output
    
    def optim(self, gradint):
        gradint = np.array(gradint)

        if len(gradint.shape) > 2:
            d_W1 = np.matmul(gradint.transpose(0,2,1), self.X_normal)
            d_W2 = np.sum(gradint, axis=1, keepdims=True)
        else:
            d_W1 = np.matmul(gradint.T, self.X_normal)
            d_W2 = np.sum(gradint, axis=0, keepdims=True)

        if len(d_W1.shape) > len(self.W1.shape) or len(d_W2.shape) > len(self.W2.shape):
            d_W1 = np.reshape(d_W1, (d_W1.shape[-2], d_W1.shape[-1]))
            d_W2 = np.reshape(d_W2, (d_W2.shape[-2], d_W2.shape[-1]))

        D_x_normal = np.matmul(gradint, self.W1)
        num = np.shape(gradint)[-1]


        if len(self.X_normal.shape) > 2:
            d_X = (1/num) * (1/(self.std + self.epsilon)) * (num * D_x_normal - np.sum(D_x_normal, axis=-1, keepdims=True)) - ( D_x_normal * np.sum(np.matmul(D_x_normal, self.X_normal.transpose(0,2,1)), axis=-1, keepdims=True))
        else:
            d_X = (1/num) * (1/(self.std + self.epsilon)) * (num * D_x_normal - np.sum(D_x_normal, axis=-1, keepdims=True)) - ( D_x_normal * np.sum(np.matmul(D_x_normal, self.X_normal.T), axis=-1, keepdims=True))

        return [d_W1, d_W2], d_X
    
    def update_param(self, parameters):
        self.W1 -= parameters[0]
        self.W2 -= parameters[1]

###////////////////---opirators---////////////////////
    
class Adam:
    def __init__(self, lerning_rate=0.01):
        self.lern_rate = lerning_rate

        self.time = 1

    def run(self, params, beta1=0.9, beta2=0.999, epsilon=1e-8):
        new_poram = []
        for poram in params:
            M = np.zeros_like(poram)
            V = np.zeros_like(poram)

            M = beta1 * M + (1-beta1) * poram
            V = beta2 * V + (1-beta2) * (poram**2)

            hat_M = M / (1-beta1 ** self.time)
            hat_V = V / (1-beta2 ** self.time)

            new_poram.append((self.lern_rate * (hat_M / (np.sqrt(hat_V) + epsilon))))
        self.time += 1
        return new_poram
    

class SVM:
    def __init__(self, lerning_rate=0.01):
        self.lern_rate = lerning_rate
    
    def run(self, params):
        new_poram = []
        for poram in params:
            new_poram.append(poram * (self.lern_rate))
        return new_poram
    


###////////////////---Nerual network builder---////////////////////


class Show:
    def __init__(self):
        self.x_values = []
        self.y_values = []
        self.num = 0

    def count(self, new_value):
        self.y_values.append(new_value)
        self.x_values.append(self.num)

        #os.system("cls")
        #print(f"loss == {self.y_values[-1]}")

        plt.xlim(0, max(self.x_values)+1)
        plt.ylim(0, 100)
        plt.scatter(self.x_values, self.y_values, color='black')
        plt.pause(0.001)
        self.num += 1
    
    def show():
        plt.show()


class NN:
    def __init__(self, layers, data, epoch, optimezator, loss, shuffle=False, batch=False, show=False):
        self.data = data
        self.optim = optimezator
        self.loss = loss

        self.map = Show()
        self.show = show

        self.batch = batch
        self.epoch = epoch
        self.shuffle = shuffle

        self.layers = layers

    def farword(self, X):
        data_layer = X
        for layer in self.layers:
            data_layer = layer.run(data_layer)
        return data_layer
    
    def backword(self, gradient):
        grad = gradient
        for layer in reversed(self.layers):
            params, grad = layer.optim(grad)

            if np.any(np.isnan(grad)):
                print(f"epoch : {i}, ")
            
            if params[0] is not None:
                new_params = self.optim.run(params)
                layer.update_param(new_params)
        return gradient

    def fit(self):
        loss_show = []

        for epoh in range(self.epoch):

            if self.shuffle:
                self.data.Shuffel()

            if self.batch == False:
                Train_batch = self.data.get_train_data()
                validation_batch = self.data.get_validation_data()
            else:
                Train_batch = self.data.get_train_batch(self.batch)
                validation_batch = self.data.get_validation_batch(self.batch)

            for batch in Train_batch:
                x_batch, y_batch = batch[0], batch[1]
                output = self.farword(x_batch)

                loss = self.loss.derivative(y_batch, output)
                real_loss = self.loss.loss(y_batch, output)
                loss_show.append(real_loss)

                if self.show:
                    self.map.count(loss_show[-1])
                    #print(f"output == {np.round(output)[-1]},  tgarget == {y_batch[-1]}, loss == {loss_show}")
                
                grid = self.backword(loss)
        return loss_show
    


class Decoder:
    def __init__(self, layers, optimezator):
        self.layers = layers
        self.optim = optimezator

    def farword(self, X, encoder_X):
        data_layer = X
        for layer in self.layers:

            try:
                if layer.masked:
                    data_layer = layer.run(data_layer, enencoder_X=encoder_X)
                else:
                    data_layer = layer.run(data_layer)

            except Exception as e:
                data_layer = layer.run(data_layer)

        return data_layer
    
    def backword(self, gradient):
        grad = gradient
        encoder_grod = None
        for layer in reversed(self.layers):

            try:
                if layer.masked:
                    params, grad, encoder_grod = layer.optim(grad)
                else:
                    params, grad = layer.optim(grad)
            except Exception as e:
                if e == 'matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 16 is different from 4)':
                    raise
                params, grad = layer.optim(grad)

            if params[0] is not None:
                new_params = self.optim.run(params)
                layer.update_param(new_params)

        return gradient, encoder_grod


class Transformer:
    def __init__(self, layers, data, optimezator, loss, epoch, shuffle=False, batch=False, show=False):
        self.data = data
        self.layers = layers
        self.optim = optimezator
        self.loss = loss
        self.epoch = epoch

        self.shuffle = shuffle

        self.show = show
        self.batch = batch
        self.map = Show()

        self.none_masked_data = None

    def farword(self, X, full_X):
        input = X
        for i, layer in enumerate(self.layers):
            if i == 0:
                self.none_masked_data = layer.farword(full_X)
            if i > 0:
                try:
                    input = layer.farword(input, self.none_masked_data)
                except Exception as e:
                    print(f"cant pass layer number : {i}")
                    raise
        return input

    def backword(self, grid):
        gradient = grid
        embed_grid = None
        for i, layer in enumerate(reversed(self.layers)):
                if i == 0:
                    gradient, embed_grid = layer.backword(gradient)
                else:
                    try:
                        gradient = layer.backword(embed_grid)
                    except Exception as e:
                        print(f"cant pass layer number : {i} reversed")
                        raise
        return gradient

    def fit(self, token_store=Embedding(toke_lib)):
        masked_data = ""
        lossing = []
        for i in range(self.epoch):

            if self.shuffle:
                self.data.Shuffel()

            if self.batch == False:
                Train_batch = self.data.get_train_data()
                validation_batch = self.data.get_validation_data()
            else:
                Train_batch = self.data.get_train_batch(self.batch)
                validation_batch = self.data.get_validation_batch(self.batch)


            for batch in Train_batch:
                dataX, dataY = batch[0], batch[1]

                for fraz in dataX:
                    split_data = fraz.strip().split()
                    masked_data = np.random.choice(["<pad>"], (np.shape(split_data)))

                    for index, element in enumerate(split_data):

                        masked_data[index] = element
                        masked_text = np.array([' '.join(map(str, masked_data.flatten()))])

                        try:
                            target = token_store.run([split_data[index+1]])
                        except Exception as e:
                            break

                        output = self.farword(masked_text, np.array([fraz]))
                        if np.any(np.isnan(output)):
                            print(f"epoch : {i}, ")

                        loss = self.loss.derivative(target[0], output)
                        lossing.append(self.loss.loss(target[0], output))

                        if self.show:
                            self.map.count(lossing[-1])

                        gradient = self.backword(loss)

np.random.seed(123)





encoder1_test = NN([Embedding(toke_lib),
            positional_encoding(),
            multi_head_attention(2),
            normalization(),
            Flatten(),
            Dense(1, 80, Relu(), flatten_befor=True),
            Dense(80, 2, softmax())],
            word_data, 10, Adam(), categorical_cross_entropy(), True, 1, show=True)
loss = encoder1_test.fit()



# encoder1 = NN([Embedding(toke_lib),
#             positional_encoding(),
#             multi_head_attention(2),
#             normalization(),
#             Flatten(),
#             Dense(1, 80, Relu(), flatten_befor=True),
#             Dense(80, 16, Relu()),
#             normalization(out=16),
#             Reshape_output(shape=(1,4,4))],
#             word_data, 10, Adam(), categorical_cross_entropy(), True, 1, show=True)

# decoder1 = Decoder([
#             Embedding(toke_lib),
#             positional_encoding(),
#             multi_head_attention(2),
#             normalization(),
#             multi_head_attention(2, non_masked=True),
#             Flatten(),
#             Dense(1, 80, Relu(), flatten_befor=True),
#             Dense(80, 16, Relu()),
#             Dense(16, 4, softmax()),
#             normalization(out=4)], Adam())

# trans_model = Transformer([encoder1,
#                            decoder1],
#                            word_data, Adam(), categorical_cross_entropy(), 30, shuffle=True, batch=3, show=True)
# trans_model.fit()













# model = NN([Grid([3,3]),
#             Conv2D([3,3], Relu(), filter_num=3),
#             poolingMax(),
#             Conv2D([3,3], Relu(), filter_num=3),
#             Flatten(),
#             Dense(1, 80, Relu(), flatten_befor=True),
#             Dense(80, 50, Relu()),
#             Dense(50, 3, softmax())],
#         data_set_photo_num_ob, 20, 5, Adam(), categorical_cross_entropy(), show=False)
# loss = model.fit()
# print(f"model : 1  \"Adam\"  ---- avrag loss : {np.average(loss)}")

# model = NN([Grid([3,3]),
            
#             Conv2D([3,3], Relu(), filter_num=3),
#             poolingMax(),
#             Conv2D([3,3], Relu(), filter_num=3),
#             Flatten(),
#             Dense(1, 80, Relu(), flatten_befor=True),
#             Dense(80, 50, Relu()),
#             Dense(50, 3, softmax())],
#             data_set_photo_num_ob, 10, Adam(), categorical_cross_entropy(), True, 25, show=True)
# loss = model.fit()
# print(f"model : 2  \"SVM\"  ---- avrag loss : {np.average(loss)}")

# model1 = NN([Conv2D([3,3], Relu(), filter_num=3),
#             poolingMax(),
#             Conv2D([3,3], Relu(), filter_num=3),
#             Flatten(),
#             Dense(1, 80, Relu(), flatten_befor=True),
#             Dense(80, 50, Relu()),
#             Dense(50, 3, softmax())],
#         data_set_photo_num_ob, 20, 5, Adam(), categorical_cross_entropy(), show=True)
# loss1 = model1.fit()
# print(f"model : 2  \"non grid data\"  ---- avrag loss : {np.average(loss1)}")

# plt.show()

# model2 = NN([Embedding(toke_lib),
#             positional_encoding(),
#             multi_head_attention(2),
#             normalization(),
#             Flatten(),
#             Dense(1, 80, Relu(), flatten_befor=True),
#             Dense(80, 2, Relu())],
#         word_data, 3, 1, Optim(), categorical_cross_entropy())
# loss2 = model2.fit()
# print(f"model : 3 \"transformer\"  ---- avrag loss : {np.average(loss2)}")


# model3 = NN([Dense(1, 80, Relu(), flatten_befor=True),
#              Dense(80, 4, Relu())],
#              clastur_data, 10, Adam(), categorical_cross_entropy(), True, 25, show=True)
# loss3 = model3.fit()

plt.show()