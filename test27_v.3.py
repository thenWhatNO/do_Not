import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import sys

def one_how(labels, num_class):
    return np.eye(num_class)[labels.astype(int)]


data_path = "data_2/object_label.csv"

df = pd.read_csv(data_path)
images= df['image']
labels = df['targ']
color = True

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
    img_array_for_show.append(cannal_up)

one_label = one_how(labels, 3)

data_set_photo_num_ob = [img_array_for_show, one_label]



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
    
    def derivative(self, x, y):
        out = self.run(x)
        return out - y
    

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
    def __init__(self, input, output, activation_func, flatten_befor=False):
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
        self.A_out = self.activation_func.run(self.Z_out)
        return self.A_out
    
    def optim(self, gradint):
         D_Z = gradint * self.activation_func.derivative(self.Z_out) # how to get the shape (10,1,3)
         D_W = np.matmul(D_Z.T, self.X)
         D_B = np.sum(D_Z, axis=0, keepdims=True)
         D_A = np.matmul(gradint, self.Wight)

         return [D_W, D_B], D_A
    
    def update_param(self, parameters):
        self.Wight -= parameters[0]
        self.Bios -= parameters[1]


class Conv2D:
    def __init__(self, kernel_size, activation_func, grid=False, grid_size=[], filter_num = 1, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel = np.random.randn(filter_num, kernel_size[0], kernel_size[1], 1)
        self.Z_out = None
        self.A_out = None
        self.activation_func = activation_func
        self.input = None
        self.grid = grid
        self.grid_size = grid_size

    def grid_spliter(self, X, H_aplit, V_split):
        self.is_grid = True
        boxes = []

        if len(np.shape(X)) == 3:
            img_in_work = np.array(img_in_work[None,:,:,:])

        batch_size, input_height, input_width, cannals_num =  np.shape(img_in_work)

        H_jump = int(input_height / H_aplit)
        V_jump = int(input_width / V_split)

        for b in range(0, batch_size):
            boxes.append([])
            for h in range(0, input_height, H_jump):
                for v in range(0, input_width, V_jump):
                    h_end, v_end = h + H_jump, v + V_jump

                    box = img_in_work[b, h:h_end, v:v_end, :]
                    boxes[b].append(box)
        
        teta = np.array(boxes).tolist()
        return teta

    def run(self, X):

        if self.grid:
            X = self.grid_spliter(X, self.grid_size[0], self.grid_size[1])


        if self.padding > 0:
            X = np.pad(X, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant').tolist()

        self.input = X

        I_B, I_H, I_W, I_C = X.shape

        O_H = (I_H - np.shape(self.kernel)[1]) // self.stride + 1
        O_W = (I_W - np.shape(self.kernel)[2]) // self.stride + 1
        
        output = np.zeros((I_B, O_H, O_W, np.shape(self.kernel)[0]))

        for y in range(0, O_H):
            for x in range(0, O_W):
                y_start, x_start = y*self.stride, x*self.stride
                y_end, x_end = y_start + np.shape(self.kernel)[1], x_start + np.shape(self.kernel)[2]
                region = X[:, y_start:y_end, x_start:x_end, :]

                if 1 != region.shape[-1]:
                    region = np.split(region, X.shape[-1], axis=3)
                    temp_out = np.zeros(output[:,y,x,:].shape)
                    for r, k in zip(region, self.kernel):
                        one = np.tensordot(r, self.kernel, axes=([1,2,3], [1,2,3]))
                        one = np.squeeze(one, axis=-2)
                        temp_out += one
                    output[:,y,x,:] = temp_out

                else:
                    output[:, y, x, :] = np.tensordot(region, self.kernel, axes=([1,2,3], [1,2,3]))

        self.Z_out = output
        self.A_out = self.activation_func.run(output)
        return self.A_out

    def optim(self, gradint):
        D_A = np.zeros_like(self.input)
        D_K = np.zeros_like(self.kernel)
        gradint = np.array(gradint)

        D_Z = self.activation_func.derivative(self.Z_out)

        for f in range(0, np.shape(self.kernel)[0]):
            for y in range(0, np.shape(gradint)[1]):
                for x in range(0, np.shape(gradint)[2]):
                    for c in range(0, np.shape(gradint)[-1]):
                        y_start, x_start = y*self.stride, x*self.stride
                        y_end, x_end = y_start + np.shape(self.kernel)[1], x_start + np.shape(self.kernel)[2]
                        
                        region = self.input[:, y_start:y_end, x_start:x_end, :]
                        teta = (gradint[:,y,x,f] * D_Z[:,y,x,f])

                        D_K[f,:,:,:] += np.sum(region[:,:,:] * teta[:,None,None,None])
                        D_A[:,y_start:y_end, x_start:x_end, :] += self.kernel[f,:,:,:] * teta[:, None,None,None]

        
        return [D_K], D_A

    def update_param(self, parameters):
        self.kernel -= parameters[0]


class poolingMax:
    def __init__(self, steps=2):
        self.X = None
        self.steps = steps
        self.X_shape = None

    def run(self, X):
        self.input = X
        self.X_shape = np.shape(X)

        I_B, I_H, I_W, I_C = self.X_shape

        O_H = (I_H - self.steps) // self.steps + 1
        O_W = (I_W - self.steps) // self.steps + 1
        
        output = np.zeros((I_B, O_H, O_W, np.shape(self.kernel)[0]))

        for h in range(0, np.shape(output)[1]):
            for w in range(0, np.shape(output)[2]):
                region = X[:, h*self.steps:h*self.steps+self.steps, w*self.steps:w*self.steps+self.steps, :]
                max_found = np.max(region)
                output[:, h, w, :] = max_found

        return output

    def optim(self, gradint):
        find = 1

        output = np.zeros_like(self.X)

        for h in range(0, np.shape(output)[1]):
            for w in range(0, np.shape(output)[2]):
                region = gradint[:, h*self.steps:h*self.steps+self.steps, w*self.steps:w*self.steps+self.steps, :]
                found_max = np.where(region == np.max(region))
                found_list = list(zip(found_max[2], found_max[3]))
                output[:, h*self.steps+found_list[0][0], w*self.steps+found_list[0][1], :] = find
                find += 1
        return [None], output
    
    def update_param(self, parameters):
        pass


class Flatten:
    def __init__(self, output_size):
        self.X = None
        self.X_shape = None
        self.output_size = output_size

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
    def __init__(self, head_num):
        self.Q_w = np.random.randn(4, 4) * np.sqrt(2. / 4)
        self.K_w = np.random.randn(4, 4) * np.sqrt(2. / 4)
        self.V_w = np.random.randn(4, 4) * np.sqrt(2. / 4)
        self.O_w = np.random.randn(4, 4) * np.sqrt(2. / 4)
        self.X = None
        self.output = None
        self.head_num = head_num

    def split_or_mix(self, X, num_heads, action):
        if action == 'split':
            batch_size, seq_length, d_model = X.shape
            depth_per_head = d_model // num_heads
            X = X.reshape(batch_size, seq_length, num_heads, depth_per_head)

            return np.transpose(X, axes=(0, 2, 1, 3))
        
        if action == 'mix':

            batch_size, num_heads, seq_length, depth_per_head = X.shape
            d_model = num_heads * depth_per_head
            X = np.transpose(X, axes=(0, 2, 1, 3))

            return X.reshape(batch_size, seq_length, d_model)

    def run(self, X):
        self.X = X
        self.D_model = np.shape(X)[-1]
        depth_per_head = self.D_model // self.head_num

        Q = np.matmul(X, self.Q_w)
        K = np.matmul(X, self.K_w)
        V = np.matmul(X, self.V_w)

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
            self.attantion_w.append(softmax(self.score[i]))
            self.attantion_out.append(np.matmul(self.attantion_w[i], self.V_heads[:, i]))

        self.attantion_out = np.stack(self.attantion_out, axis=1)
        self.combain_out = self.split_or_mix(self.attantion_out, self.head_num, "mix")

        O = np.dot(self.combain_out, self.O_w)
        return O
    
    def optim(self, gradint):
        D_O = np.matmul(gradint, self.O_w.swapaxes(-1,-2))
        D_wo = np.matmul(self.combain_out.reshape(-1, self.D_model).T, D_O.reshape(-1, self.D_model))
        split_D_O = self.split_or_mix(D_O, self.head_num, "split")
        d_k = []
        d_q = []
        d_v = []

        for i in range(self.head_num):
            D_V_head = np.matmul(self.attantion_w[i], split_D_O[:,i,:,:])
            d_v.append(D_V_head)

            d_attantion_w = np.matmul(split_D_O[:,i,:,:], self.V_heads[:,i].swapaxes(-1,-2))
            d_scale = d_attantion_w * softmax.derivative(self.attantion_w[i])

            D_Q_head = np.matmul(d_scale, self.K_heads[:,i])
            D_k_head = np.matmul(d_scale, self.Q_heads[:,i])

            d_q.append(D_Q_head)
            d_k.append(D_k_head)

        d_k = self.split_or_mix(np.array(d_k).transpose(1,0,2,3), self.head_num, "mix")
        d_q = self.split_or_mix(np.array(d_q).transpose(1,0,2,3), self.head_num, "mix")
        d_v = self.split_or_mix(np.array(d_v).transpose(1,0,2,3), self.head_num, "mix")
    
        d_A = np.matmul(d_q, self.Q_w.T) + np.matmul(d_k, self.K_w.T) + np.matmul(d_v, self.V_w)

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
    def __init__(self, link):
        self.link = link
        self.tabel = pd.read_csv(link)

    def add_new_words(self, words):
        for word in words:
            if not (self.tabel["word"] == word).any():
                token = np.random.randn(4)
                token = np.array2string(token, separator=',')
                self.tabel.loc[len(self.tabel)] = [word, self.tabel['id'].iloc[-1]+1, token]

        self.tabel.to_csv('tokens.csv', index=False)

        print("the program stap work becouse of new data get added to the tokkins data \n reset the program to keep work")
        sys.exit()

    def run(self, X):
        self.output = []

        for sentens in X:
            input_word = sentens.strip().split()
            row = self.tabel[self.tabel['word'].isin(input_word)]
            word = row["word"]

            if len(word) < len(input_word):
                self.add_new_words(input_word)

            token = row["token"]
            ids = row["id"]
            token_correct = np.array([np.genfromtxt([row.strip("[]")], delimiter=",") for row in token])
            self.output.append(token_correct)
        return self.output
    
    def optim(self, gradint):
        output = []

        self.output -= np.array(gradint) * 0.01

        for i, sentens in enumerate(self.X):
            input_word = sentens.strip().split()
            row = self.tabel[self.tabel['word'].isin(input_word)]

            token_data = self.output[i]
            id = row["id"]

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
    def __init__(self, epsilon=1e-6):
        self.W1 = np.random.randn(4, 4) * np.sqrt(2. / 4)
        self.W2 = np.random.randn(4, 4) * np.sqrt(2. / 4)
        self.epsilon = epsilon
        self.X = None

    def run(self, X):
        self.X = X
        mean_X = np.mean(X)
        self.std = np.std(X)
        self.X_normal = (X - mean_X) / (self.std + self.epsilon)

        output = np.dot(self.X_normal, self.W1) + self.W2
        return output
    
    def optim(self, gradint):
        d_W1 = np.dot(gradint, self.X_normal)
        d_W2 = gradint

        D_x_normal = np.matmul(gradint, self.W1)
        num = np.shape(gradint)[-1]

        d_X = (1/num) * (1/(self.std + self.epsilon)) * (num * D_x_normal - np.sum(D_x_normal, axis=-1, keepdims=True)) - ( D_x_normal * np.sum(np.dot(D_x_normal, self.X_normal), axis=-1, keepdims=True))

        return [d_W1, d_W2], d_X
    
    def update_param(self, parameters):
        self.W1 = parameters[0]
        self.W2 = parameters[1]

###////////////////---opirators---////////////////////
    


class Optim:
    def __init__(self, lerning_rate=0.01):
        self.lern_rate = lerning_rate

        self.time = 1

    def Adam(self, params, beta1=0.9, beta2=0.999, epsilon=1e-8):
        new_poram = []
        for poram in params:
            M = np.zeros_like(poram)
            V = np.zeros_like(poram)

            M = beta1 * M + (1-beta1) * poram
            V = beta2 * V + (1-beta2) * (poram**2)

            hat_M = M / (1-beta1 ** self.time)
            hat_V = V / (1-beta2 ** self.time)

            new_poram.append((self.lern_rate * hat_M / (np.sqrt(hat_V) + epsilon)))
        self.time += 1
        return new_poram
    
    def SVM(self, params):
        new_poram = []
        for poram in params:
            new_poram.append(poram * self.lern_rate)
        return new_poram
    


###////////////////---opiratoNerual network builder---////////////////////


class Show:
    def __init__(self):
        self.x_values = []
        self.y_values = []
        self.num = 0

    def count(self, new_value):
        self.y_values.append(new_value)
        self.x_values.append(self.num)

        #os.system("cls")
        print(f"loss == {self.y_values[-1]}")

        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.scatter(self.x_values, self.y_values, color='black')
        plt.pause(0.001)
        self.num += 1
    
    def show():
        plt.show()


class NN:
    def __init__(self, data, batch, epoch, optimezator, loss):
        self.X_data, self.Y_data = data[0], data[1]
        self.optim = optimezator
        self.loss = loss

        self.map = Show()

        self.batch = batch
        self.epoch = epoch

        self.layers = [
            Conv2D([3,3], Relu(), filter_num=3),
            Conv2D([3,3], Relu(), filter_num=3),
            Conv2D([3,3], Relu(), filter_num=3),
            Flatten(20),
            Dense(20, 2, Relu(), flatten_befor=True),
            Dense(2, 4, Relu()),
            Dense(4, 3, Sigmoid())
        ]

    def Shuffel(self, x, y):
        assert len(y) == len(x)
        index = np.arange(len(x))
        np.random.shuffle(index)

        x = np.array(x)[index]
        y = np.array(y)[index]
        return x, y

    def split(self, x_data, y_data, split_procent=0.8):
        split_use = int(len(x_data) * split_procent)

        val_data_x = x_data[split_use:]
        val_data_y = y_data[split_use:]
        train_data_x = x_data[:split_use]
        train_data_y = y_data[:split_use]

        return val_data_x, val_data_y, train_data_x, train_data_y

    def farword(self, X):
        data_layer = X
        for layer in self.layers:
            data_layer = layer.run(data_layer)
        return data_layer
    
    def backword(self, gradient):
        grad = gradient
        for layer in reversed(self.layers):
            params, grad = layer.optim(grad)
            
            if params[0] is not None:
                new_params = self.optim.Adam(params)
                layer.update_param(new_params)

    def fit(self):

        self.X_data, self.Y_data = self.Shuffel(self.X_data, self.Y_data)
        val_data_x, val_data_y, train_data_x, train_data_y = self.split(self.X_data, self.Y_data)

        for epoh in range(self.epoch):
            for batch in range(0, len(train_data_x), self.batch):

                x_batch = train_data_x[batch:batch + self.batch]
                y_batch = train_data_y[batch:batch + self.batch]

                output = self.farword(x_batch)

                loss = self.loss.derivative(y_batch, output)
                loss_show = self.loss.loss(y_batch, output)

                self.map.count(loss_show)

                print(f"output == {np.round(output)[-1]},  tgarget == {y_batch[-1]}")
                
                self.backword(loss)

        

plt.show()

model = NN(data_set_photo_num_ob, 10, 10, Optim(), categorical_cross_entropy())
model.fit()