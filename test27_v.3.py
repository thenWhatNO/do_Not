import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import sys

toke_lib = "tokens.csv"

def one_how(labels, num_class):
    return np.eye(num_class)[labels.astype(int)]

##################### ---------- token/text data ------------

data_path = "data_2/words_label.csv"

df = pd.read_csv(data_path)
sentens= df['sentenc'].tolist()
labels = np.array(df['targ'])

eye_labols = one_how(labels, 2)

word_data = [sentens, eye_labols]

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
        if self.activation_func != None:
            self.A_out = self.activation_func.run(self.Z_out)
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
        self.new_shape = shape

    def run(self, X):
        try:
            return np.reshape(X, self.new_shape)
        except Exception as e:
            raise

    def optim(self, gradint):
        pass
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
                            D_A[b, y:y+self.kernel.shape[0],   x:x+self.kernel.shape[1]] += opa[:,:,None]

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
            batch_size, seq_length, d_model = X.shape
            depth_per_head = d_model // num_heads
            X = X.reshape(batch_size, seq_length, num_heads, depth_per_head)

            return np.transpose(X, axes=(0, 2, 1, 3))
        
        if action == 'mix':

            batch_size, num_heads, seq_length, depth_per_head = X.shape
            d_model = num_heads * depth_per_head
            X = np.transpose(X, axes=(0, 2, 1, 3))

            return X.reshape(batch_size, seq_length, d_model)

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
        test1 = self.combain_out.reshape(-1, self.D_model).T
        test2 = D_O.reshape(-1, self.D_model)
        D_wo = np.matmul(self.combain_out.reshape(-1, self.D_model).T, D_O.reshape(-1, self.D_model))
        split_D_O = self.split_or_mix(D_O, self.head_num, "split")
        d_k = []
        d_q = []
        d_v = []

        for i in range(self.head_num):
            D_V_head = np.matmul(self.attantion_w[i], split_D_O[:,i,:,:])
            d_v.append(D_V_head)

            d_attantion_w = np.matmul(split_D_O[:,i,:,:], self.V_heads[:,i].swapaxes(-1,-2))
            d_scale = d_attantion_w * softmax.derivative(None, self.attantion_w[i])

            D_Q_head = np.matmul(d_scale, self.K_heads[:,i])
            D_k_head = np.matmul(d_scale, self.Q_heads[:,i])

            d_q.append(D_Q_head)
            d_k.append(D_k_head)

        d_k = self.split_or_mix(np.array(d_k).transpose(1,0,2,3), self.head_num, "mix")
        d_q = self.split_or_mix(np.array(d_q).transpose(1,0,2,3), self.head_num, "mix")
        d_v = self.split_or_mix(np.array(d_v).transpose(1,0,2,3), self.head_num, "mix")
    
        d_A = np.matmul(d_q, self.Q_w.T) + np.matmul(d_k, self.K_w.T) + np.matmul(d_v, self.V_w)

        d_k = np.mean(d_k, axis=0)
        d_q = np.mean(d_q, axis=0)
        d_v = np.mean(d_v, axis=0)
        d_A = np.mean(d_A, axis=0)

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

        for i, sentens in enumerate(self.X):
            input_word = sentens.strip().split()
            row = self.tabel[self.tabel['word'].isin(input_word)]

            token_data = self.output[i]
            id = row["id"].tolist()

            for i, word in enumerate(input_word):
                match_row = self.tabel.loc[self.tabel['word'].isin([word]), "token"]
                if not match_row.empty:
                    update = np.array2string(token_data[i], separator=',')
                    self.tabel.at[id[i], "token"] = update
        if self.updata_token:
            self.tabel.to_csv(self.link, index=False)
        return [None], output
    
    def update_param(self, parameters):
        pass


class normalization:
    def __init__(self, out=None, epsilon=1e-6):
        self.W1 = None
        self.W2 = None
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

        if self.W1 == None or self.W2:
            self.creat_param(X.shape[-2], X.shape[-1]) if self.out == None else self.creat_param(X.shape[-2], self.out)

        output = np.matmul(self.X_normal, self.W1) + self.W2
        return output
    
    def optim(self, gradint):
        gradint = np.array(gradint)

        if len(gradint) > 2:
            d_W1 = np.matmul(gradint, self.X_normal)
            d_W2 = np.sum(gradint, axis=1, keepdims=True)
            d_W1 = np.mean(d_W1 , axis=0)
            d_W2 = np.mean(d_W2 , axis=0)
        else:
            d_W1 = np.matmul(gradint, self.X_normal)
            d_W2 = np.sum(gradint, axis=0, keepdims=True)

        D_x_normal = np.matmul(gradint, self.W1)
        num = np.shape(gradint)[-1]

        d_X = (1/num) * (1/(self.std + self.epsilon)) * (num * D_x_normal - np.sum(D_x_normal, axis=-1, keepdims=True)) - ( D_x_normal * np.sum(np.dot(D_x_normal, self.X_normal), axis=-1, keepdims=True))

        return [d_W1, d_W2], d_X
    
    def update_param(self, parameters):
        self.W1 -= parameters[0]
        self.W2 -= parameters[1]

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

            new_poram.append((self.lern_rate * (hat_M / (np.sqrt(hat_V) + epsilon))))
        self.time += 1
        return new_poram
    
    def SVM(self, params):
        new_poram = []
        for poram in params:
            new_poram.append(poram * (self.lern_rate))
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
        #print(f"loss == {self.y_values[-1]}")

        plt.xlim(0, max(self.x_values)+1)
        plt.ylim(0, 100)
        plt.scatter(self.x_values, self.y_values, color='black')
        plt.pause(0.001)
        self.num += 1
    
    def show():
        plt.show()


class NN:
    def __init__(self, layers,  data, batch, epoch, optimezator, loss, show=False):
        self.X_data, self.Y_data = data[0], data[1]
        self.optim = optimezator
        self.loss = loss

        self.map = Show()
        self.show = show

        self.batch = batch
        self.epoch = epoch

        self.layers = layers

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
        return gradient

    def fit(self):

        self.X_data, self.Y_data = self.Shuffel(self.X_data, self.Y_data)
        val_data_x, val_data_y, train_data_x, train_data_y = self.split(self.X_data, self.Y_data)
        loss_show = []

        for epoh in range(self.epoch):
            for batch in range(0, len(train_data_x), self.batch):

                x_batch = train_data_x[batch:batch + self.batch]
                y_batch = train_data_y[batch:batch + self.batch]

                output = self.farword(x_batch)

                loss = self.loss.derivative(y_batch, output)
                loss_show.append(self.loss.loss(y_batch, output))

                if self.show:
                    self.map.count(loss_show[-1])
                    #print(f"output == {np.round(output)[-1]},  tgarget == {y_batch[-1]}, loss == {loss_show}")
                
                grid = self.backword(loss)
                plt.show()
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
            except Exception as e:
                data_layer = layer.run(data_layer)

        return data_layer
    
    def backword(self, gradient):
        grad = gradient
        encoder_grod = None
        for layer in reversed(self.layers):

            try:
                params, grad, encoder_grod = layer.optim(grad)
            except Exception as e:
                params, grad = layer.optim(grad)

            if params[0] is not None:
                new_params = self.optim.Adam(params)
                layer.update_param(new_params)

        return gradient, encoder_grod
    



class Transformer:
    def __init__(self, layers, data, optimezator, loss, epoch):
        self.X_data, self.Y_data = data[0], data[1]
        self.layers = layers
        self.optim = optimezator
        self.loss = loss
        self.epoch = epoch

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
        return input

    def backword(self, grid):
        gradient = grid
        embed_grid = None
        for i, layer in enumerate(reversed(self.layers)):
            try:
                gradient, embed_grid = layer.backword(gradient)
            except Exception as e:
                if "too many positional arguments" in str(e): 
                    gradient = layer.backword(gradient)
                else:
                    raise
        return gradient

    def fit(self, token_store=Embedding(toke_lib)):
        masked_data = ""
        lossing = []
        for i in range(self.epoch):
            for ind, full_data in enumerate(self.X_data):
                split_data = full_data.strip().split()
                masked_data = np.random.choice(["<pad>"], (np.shape(split_data)))
                for index, element in enumerate(split_data):
                    masked_data[index] = element
                    masked_text = np.array([' '.join(map(str, masked_data.flatten()))])

                    output = self.farword(masked_text, np.array([full_data]))

                    target = token_store.run([split_data[index+1]])

                    loss = self.loss.derivative(target[0], output)
                    lossing.append(self.loss.loss(target[0], output))
                    gradient = self.backword(loss)
        

# model = NN([Grid([3,3]),
#             Conv2D([3,3], Relu(), filter_num=3),
#             poolingMax(),
#             Conv2D([3,3], Relu(), filter_num=3),
#             Flatten(),
#             Dense(1, 80, Relu(), flatten_befor=True),
#             Dense(80, 50, Relu()),
#             Dense(50, 3, softmax())],
#         data_set_photo_num_ob, 20, 5, Optim(), categorical_cross_entropy())
# loss = model.fit()
# print(f"model : 1  \"grid data\"  ---- avrag loss : {np.average(loss)}")

# model1 = NN([Conv2D([3,3], Relu(), filter_num=3),
#             poolingMax(),
#             Conv2D([3,3], Relu(), filter_num=3),
#             Flatten(),
#             Dense(1, 80, Relu(), flatten_befor=True),
#             Dense(80, 50, Relu()),
#             Dense(50, 3, softmax())],
#         data_set_photo_num_ob, 20, 5, Optim(), categorical_cross_entropy())
# loss1 = model1.fit()
# print(f"model : 2  \"non grid data\"  ---- avrag loss : {np.average(loss1)}")

model2 = NN([Embedding(toke_lib),
            positional_encoding(),
            multi_head_attention(2),
            normalization(),
            Flatten(),
            Dense(1, 80, Relu(), flatten_befor=True),
            Dense(80, 2, Relu())],
        word_data, 3, 1, Optim(), categorical_cross_entropy())
loss2 = model2.fit()
print(f"model : 3 \"transformer\"  ---- avrag loss : {np.average(loss2)}")


encoder1 = NN([Embedding(toke_lib),
            positional_encoding(),
            multi_head_attention(2),
            normalization(),
            Flatten(),
            Dense(1, 80, Relu(), flatten_befor=True),
            Dense(80, 16, Relu()),
            normalization(out=16),
            Reshape_output(shape=(1,4,4))],
        word_data, 20, 1, Optim(), categorical_cross_entropy())

decoder1 = Decoder([
            Embedding(toke_lib),
            positional_encoding(),
            multi_head_attention(2),
            normalization(),
            multi_head_attention(2, non_masked=True),
            Flatten(),
            Dense(1, 80, Relu(), flatten_befor=True),
            Dense(80, 16, Relu()),
            Dense(16, 4, softmax()),
            normalization(out=4)], Optim())

trans_model = Transformer([encoder1,
                           decoder1],
                           word_data, Optim(), categorical_cross_entropy(), 5)
trans_model.fit()
