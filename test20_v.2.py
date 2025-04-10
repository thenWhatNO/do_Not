import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import sys

toke_lib = "tokens.csv"
tokins = pd.read_csv(toke_lib)



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

# one_label = one_how(labels, 2)

# data_set_photo = [img_array_for_show, one_label]

#/////////////////////////////////////////////

data_path = "data_2/words_label.csv"

df = pd.read_csv(data_path)
sentens= df['sentenc'].tolist()
labels = np.array(df['targ'])

eye_labols = one_how(labels, 2)

word_data = [sentens, eye_labols]

#////////////////////////////////////num ob data

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

#////////////////////////////////////

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

# one_label = one_how(labels, 26)

# data_latine_digets = [img_array_for_show, one_label]


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

#plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, cmap='viridis')

data_set1 = [data_x, data_y_one_hot]


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

        self.is_grid = False
        self.optim_time = False
        self.atantion_optim = False
        self.Creat_time = True
        self.is_convFirst = False
        self.is_word_data = False
        self.conv_optim = False

        self.batch_num = 1
        self.filter = 1
        self.channal = 1

        self.count_layers_num = 0

        self.Wight = []
        self.Bias = []
        self.kernel = []

        self.Output = []
        self.Z_output = []

    def Layers(self):
        if self.optim_time:
            self.Dense(None, None, "softmax")
            self.Dense(None, None, "relu")
            self.Dense(None, None, "relu")
            self.Flatten()
            self.conv2d(3, "relu")
            self.optim_time = False
            return
        self.conv2d(3, "relu")
        out_shape = self.Flatten()
        self.Dense(out_shape, 40 , "relu")
        self.Dense(40, 20, "relu")
        self.Dense(20 , 2, "softmax")

    def add_output_layer(self):
        self.Output = []
        self.Z_output = []

        for i in range(self.count_layers_num+1):
            self.Output.append([])
            self.Z_output.append([])
        self.Z_output.pop()

    def Creat(self):
        if self.is_grid:
            image = self.grid_spliter(self.X_data[0], 3, 3)
            self.test_img = np.zeros((np.shape(image))).tolist()
            self.is_grid = False

        if self.is_convFirst:
            self.test_img = [[np.zeros((np.shape(self.X_data[0]))).tolist()]]

        self.count_layers_num = 0

        self.optim_time = False
        self.Creat_time = True
        self.Layers()
        self.Creat_time = False

    def Creat_param(self, neruals, weithg, return_to=False):
        np.random.seed(1)
        wight = np.random.randn(neruals, weithg) * np.sqrt(2. / weithg)
        bias = np.random.randn(1, neruals)

        if return_to:
            return wight, bias

        self.Wight.append(wight.tolist())
        self.Bias.append(bias.tolist())

    def creat_kernel(self, size, filters):

        filter_and_kernel = []
        for i in range(filters):
            kernel = np.random.randn(size, size, self.channal).tolist()

            filter_and_kernel.append(kernel)

        self.kernel.append(filter_and_kernel)

        self.Wight.append([0])
        self.Bias.append([0])

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
    
    def multi_head_attention(self, head_num): #V

        if self.Creat_time:

            self.count_layers_num += 1

            wight_bias = {
                "Q":[],
                "K":[],
                "V":[],
                "O":[]
            }

            for i in wight_bias:
                wight, bios = self.Creat_param(4, 4, True)
                wight_bias[i] = wight

            self.Wight.append(wight_bias)
            self.Bias.append([0])
            self.kernel.append([0])

            return 
        

        if not self.optim_time:
            self.atantion_data = np.array(self.Output[self.on_this])

        X_data = np.array(self.atantion_data)

        d_model = np.shape(X_data)[-1]
        depth_per_head = d_model // head_num

        Q = np.matmul(X_data, self.Wight[self.on_this]["Q"])
        K = np.matmul(X_data, self.Wight[self.on_this]["K"])
        V = np.matmul(X_data, self.Wight[self.on_this]["V"])

        Q_heads = self.split_or_mix(Q, head_num, "split")
        K_heads = self.split_or_mix(K, head_num, "split")
        V_heads = self.split_or_mix(V, head_num, "split")

        attention_outputs = []

        scores = []
        scaled_scores = []
        self.attention_weights = []
        attention_output = []

        scaling_factor = np.sqrt(depth_per_head)
        for i in range(head_num):
            scores.append(np.matmul(Q_heads[:, i], K_heads[:, i].transpose(0, 2, 1)) / scaling_factor)
            self.attention_weights.append(softmax(scores[i]))
            attention_output.append(np.matmul(self.attention_weights[i], V_heads[:, i]))

        attention_outputs = np.stack(attention_output, axis=1)
        self.combined_output = self.split_or_mix(attention_outputs, head_num, 'mix') 

        O = np.dot(self.combined_output, self.Wight[self.on_this]["O"])

        if not self.optim_time:
            self.on_this += 1
            self.Output[self.on_this] = O.tolist()
            return
             
        if self.optim_time:
            gridy = self.Output_drev[-1]
            
            d_O = np.matmul(gridy, self.Wight[self.on_this]["O"].swapaxes(-1, -2))
            d_Wo = np.matmul(self.combined_output.reshape(-1, d_model).T, d_O.reshape(-1, d_model))

            split_D_O = self.split_or_mix(d_O, head_num, "split")

            d_K = []
            d_Q = []
            d_V = []

            for i in range(head_num):
                d_V_head = np.matmul(self.attention_weights[i], split_D_O[:,i,:,:])
                d_V.append(d_V_head)
                
                d_attention_weights = np.matmul(split_D_O[:,i,:,:], V_heads[:, i].swapaxes(-1, -2))
                d_scaled_scores = d_attention_weights * softmax_derivative(self.attention_weights[i])

                d_Q_head = np.matmul(d_scaled_scores, K_heads[:,i])
                d_K_head = np.matmul(d_scaled_scores, Q_heads[:,i])

                d_Q.append(d_Q_head)
                d_K.append(d_K_head)

            d_K = self.split_or_mix(np.array(d_K).transpose(1,0,2,3), head_num, "mix")
            d_Q = self.split_or_mix(np.array(d_Q).transpose(1,0,2,3), head_num, "mix")
            d_V = self.split_or_mix(np.array(d_V).transpose(1,0,2,3), head_num, "mix")

            d_Wq = np.matmul(X_data.reshape(-1, d_model).T, d_Q.reshape(-1, d_model))
            d_Wk = np.matmul(X_data.reshape(-1, d_model).T, d_K.reshape(-1, d_model))
            d_Wv = np.matmul(X_data.reshape(-1, d_model).T, d_V.reshape(-1, d_model))

            self.d_atantion_W = {
                "Q":d_Wq,
                "K":d_Wk,
                "V":d_Wv,
                "O":d_Wo
            }

            d_A = np.matmul(d_Q, self.Wight[self.on_this]["Q"].T) + np.matmul(d_K, self.Wight[self.on_this]["K"].T) + np.matmul(d_V, self.Wight[self.on_this]["V"].T)

            self.atantion_optim = True
            self.optim(self.optim_type)
            self.on_this -= 1
            self.atantion_optim = False
        
    
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
    
    def positional_encoding(self):

        if self.Creat_time:
            
            self.is_word_data = True

            self.count_layers_num += 1

            self.Wight.append([0])
            self.Bias.append([0])
            self.kernel.append([0])

            return 
        
        x_data = self.Output[self.on_this]
        word_num, token_num = np.shape(x_data)[-2], np.shape(x_data)[-1] 

        PE = np.zeros((word_num, token_num))
        for pos in range(word_num):
            for i in range(0, token_num, 2):
                PE[pos, i] = np.sin(pos / (10000 ** (i / token_num)))
                if i + 1 < token_num:
                    PE[pos, i + 1] = np.cos(pos / (10000 ** (i / token_num)))

        outy = np.array(x_data)
        out = (outy + PE).tolist()

        if not self.optim_time:
            self.on_this += 1
            self.Output[self.on_this] = out
            return

        self.Output_drev.append(self.Output_drev[-1])

    def add_new_words(self, words):
        for word in words:
            if not (tokins["word"] == word).any():
                token = np.random.randn(4)
                token = np.array2string(token, separator=',')
                tokins.loc[len(tokins)] = [word, tokins['id'].iloc[-1]+1, token]

        tokins.to_csv('tokens.csv', index=False)

        print("the program stap work becouse of new data get added to the tokkins data \n reset the program to keep work")
        sys.exit()


    def normalization(self, W1, W2, epsilon=1e-6):
        if self.Creat_time:
            self.count_layers_num += 1
            self.Creat_param(W1, W2) 
            self.kernel.append([0])
            return 

        if not self.optim_time:
            input = self.Output[self.on_this]
        mean_x = np.mean(input)
        std = np.std(input)
        X_normal = (input - mean_x) / (std + epsilon)

        output = np.dot(X_normal, self.Wight[self.on_this]) + self.Bias[self.on_this]

        self.on_this += 1
        self.Output[self.on_this] = X_normal

        if self.optim_time:
            
            D_A = np.array(self.Output_drev[-1])

            self.W_D.append(np.dot(D_A, X_normal))
            self.B_D.append(D_A)

            D_Xnorm = np.matmul(D_A, self.Wight[self.on_this].T)

            num = D_A.shape[-1]

            D_X = (1/num) * (1 / (std + epsilon)) * (num * D_Xnorm - np.sum(D_Xnorm, axis=-1, keepdims=True)) - (X_normal * np.sum(np.dot(D_Xnorm, X_normal), axis=-1, keepdims=True))

            self.Output_drev.append(D_X)
            self.optim(self.optim_type)
            self.on_this -= 1


    def Embedding(self):

        if self.Creat_time:
            self.count_layers_num += 1

            self.Wight.append([0])
            self.Bias.append([0])
            self.kernel.append([0])

            return 

        input_seq = self.Output[0]


        self.tokinze = []
        for sentens in input_seq:
            input_words = sentens.strip().split()
            
            row = tokins[tokins['word'].isin(input_words)]
            worly = row["word"].tolist()
            
            if len(worly) < len(input_words):  
                self.add_new_words(input_words)
                return []

            token = row["token"].tolist()
#           token_corect = np.array([np.fromstring(rop.strip("[]"), sep=" ")for rop in token])
            token_corect = np.array([np.genfromtxt([row.strip("[]")], delimiter=",") for row in token])
            ids = row["id"].tolist()
            self.tokinze.append(token_corect.tolist())

        self.on_this += 1
        self.Output[self.on_this] = self.tokinze

        if self.optim_time:
            D_A = self.Output_drev[-1]
            tokenis = self.Output[1]
            
            #self.tokinze = np.array([np.fromstring(np.array(rop).strip("[]"), sep=" ")for rop in self.tokinze])
        
            tokenis -= np.array(D_A) * 0.01 # not want to multiply

            for i, sentens in enumerate(self.Output[0]):
                
                input_words = sentens.strip().split() 
                row = tokins[tokins['word'].isin(input_words)]

                tokien_for_data = tokenis[i]
                idis = row["id"].tolist()
                
                for i, wordy in enumerate(input_words):
                    matching_rows = tokins.loc[tokins['word'].isin([wordy]), "token"]
                    if not matching_rows.empty:
                        update_token =  np.array2string(tokien_for_data[i], separator=',')
                        tokins.at[idis[i], "token"] = update_token

            tokins.to_csv('tokens.csv', index=False)  
            return

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

    def conv2d(self, kernel_size, activition_func, filter_num = 1, stride=1, padding=0):

        if self.Creat_time:
            self.count_layers_num += 1
            self.is_convFirst = True
            self.kernel_org_shape = 0
            self.creat_kernel(kernel_size, filter_num)
            self.conv_optim = False

            input_image = np.array(self.test_img)

            filters, kernel_height, kernel_width, _ = np.shape(self.kernel[self.on_this])

            new_shape = self.output_img_shapere(input_image, stride, kernel_height, kernel_width, filters)

            output_test = np.zeros_like((new_shape)).tolist()
            self.test_img = output_test
            self.on_this += 1
            return
                
        input_image = np.array(self.Output[self.on_this])
        kernel_for_work = np.array(self.kernel[self.on_this])
        kernel_for_work = kernel_for_work[None,:,:,:,:] # this for the shape testing in grid status

        if self.optim_time:
            self.kernel_D.append([])
            _, filter, kernel_height, kernel_width, channals_k= np.shape(kernel_for_work)
            batch_size, _, gradient_height, gradient_width, channals_c = np.shape(self.Output_drev[-1])

            activ_drev = np.array(self.Activeit(self.Z_output[self.on_this+1], activition_func))

            A_D_out = np.array(self.Output_drev[-1])

            D_A = np.zeros_like(input_image)
            D_K = np.zeros_like(self.kernel[self.on_this])

            for f in range(0, filter):
                for y in range(0, gradient_height):
                    for x in range(0, gradient_width):
                        for c in range(0, input_image.shape[-1]):
                            h_start, w_start = y * stride, x * stride
                            h_end, w_end = h_start + kernel_height, w_start + kernel_width

                            region = input_image[:, :,h_start:h_end,w_start:w_end, c]
                            teta = (A_D_out[:,:,y,x,f] * activ_drev[:,:,y,x,f])

                            D_K[f, :, :, :] += np.sum(region[:,:,:,:,None] * teta[:,:,None,None,None], axis=(0,1))
                            D_A[:, :, h_start:h_end, w_start:w_end, :] += kernel_for_work[:, f, :, :, :] * teta[:, :, None, None, None]
                                                                                        # 1  3  3  3  3         
            self.kernel_D[-1] = D_K.tolist()
            teta = D_A.tolist()
            self.Output_drev.append(teta)
            
            self.conv_optim = True
            self.optim(self.optim_type)
            self.on_this -= 1
            self.conv_optim = False
            return
        
        #?/////////////////////////////////////// loss

        filters, kernel_height, kernel_width, cannals= np.shape(self.kernel[self.on_this])

        if padding > 0:
            input_image = np.pad(self.Output[self.on_this][-1], ((padding, padding), (padding, padding)), mode='constant').tolist()
    
        output = np.array(self.output_img_shapere(input_image, stride, kernel_height, kernel_width, filters))
        _, _, output_height, output_width, channels_img = np.shape(output)

        for y in range(0, output_height):
            for x in range(0, output_width):
                h_start, w_start = y * stride, x * stride
                h_end, w_end = h_start + kernel_height, w_start + kernel_width
                region = input_image[:, :, h_start:h_end, w_start:w_end, :]

                if 1 != region.shape[-1]:
                    region = np.split(region, input_image.shape[-1], axis=4)

                    temp_output = np.zeros(output[:, :, y, x, :].shape) 

                    for r, k in zip(region, kernel_for_work):
                        helper = np.tensordot(r, kernel_for_work, axes=([2, 3, 4], [2, 3, 4]))
                        helper = np.squeeze(helper, axis=-2)
                        temp_output += helper

                    output[:, :, y, x, :] = temp_output

                else:
                    test =  np.tensordot(region, kernel_for_work, axes=([2, 3, 4], [2, 3, 4]))
                    alpha = np.squeeze(test, axis=-2)
                    output[:, :, y, x, :] = alpha

        self.on_this += 1
        self.Z_output[self.on_this] = output.tolist()
        A = self.Activeit(np.array(self.Z_output[self.on_this]), activition_func)
        self.Output[self.on_this] = A

    def poolingMax(self, steps=2):

        if self.Creat_time:
            self.count_layers_num += 1

            self.Wight.append([0])
            self.Bias.append([0])
            self.kernel.append([0])

            test_img = self.test_img
            _,_,_,cannal = np.shape(test_img)

            new_shape = self.output_img_shapere(test_img, steps, steps, steps, cannal)

            out_test = np.zeros_like((new_shape)).tolist()
            self.test_img = out_test

            self.on_this += 1

            return

        if self.optim_time:
            self.poolimgMax_drev()
            self.on_this -= 1
        
            return

        working_img = np.array(self.Output[self.on_this])

        _, _, _, _, cannal = working_img.shape

        new_shape_W = self.output_img_shapere(working_img, steps, steps, steps, cannal)

        _, _, output_height, output_width, _ = new_shape_W.shape

        output_image = np.zeros_like((new_shape_W))

        for y in range(0, output_height):
            for x in range(0, output_width):
                region = working_img[:, :, y*steps:y*steps+steps, x*steps:x*steps+steps, :]
                opa = np.max(region)
                output_image[:, :, y, x, :] = opa

        self.on_this += 1
        
        teta = output_image.tolist()
        self.Output[self.on_this] = teta

    def poolimgMax_drev(self, steps=2):

        working_img = np.array(self.Output_drev[-1])

        find = 1
        batch, grid, inputH, input_W, cannal = working_img.shape

        new_shape = self.output_img_shapere(working_img, steps, steps, steps, cannal)

        _, _, output_height, output_width, _ = new_shape.shape

        output_image = np.zeros((batch, grid, inputH, input_W, cannal))

        for y in range(0, output_height):
            for x in range(0, output_width):
                region = working_img[:, :, y*steps:y*steps+steps, x*steps:x*steps+steps, :]
                maxy = np.max(region)
                wer = np.where(region == maxy)
                wer_list = list(zip(wer[2], wer[3]))
                output_image[:, :, y*steps+wer_list[0][0], x*steps+wer_list[0][1], :] = find
                find += 1
        
        teta = output_image.tolist()
        self.Output_drev.append(teta)

    def output_img_shapere(self, input_image, stride, kernel_height, kernel_width, filter):

        input_image = np.array(input_image)

        batch_size, grid, input_height, input_width, cannals_num = input_image.shape

        output_height = (input_height - kernel_height) // stride + 1
        output_width = (input_width - kernel_width) // stride + 1

        shape_new = (batch_size, output_height, output_width, filter)

        if self.is_grid:
            shape_new = (batch_size, grid, output_height, output_width, filter)

        output = np.zeros(shape_new)[:,None]

        return output
    
    def grid_spliter(self, X, H_aplit, V_split):
        self.is_grid = True

        boxes = []

        img_in_work = np.array(X)

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

    def Flatten(self):
        if self.Creat_time:
            self.count_layers_num += 1

            self.Wight.append([0])
            self.Bias.append([0])

            if self.is_convFirst:
                self.kernel_org_shape = np.shape(self.test_img)
                out_of = np.array(self.test_img).reshape(1, -1).tolist()

            if self.is_word_data:
                self.kernel_org_shape = (4,4)
                out_of = 4*4
                return out_of

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
        if self.is_convFirst==False and self.is_word_data==False:
            X = X.reshape(-1, 1).T
        self.on_this = 0

        X = np.array(X)
        if self.is_grid == False and self.is_word_data==False:
            X = X[:,None]

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
            if self.atantion_optim:
                self.SVM_optim_atantion(lerning_rate)
                return
            self.SVM_optim_conv(lerning_rate) if self.conv_optim == True else self.SVM_optim(lerning_rate)
        if optimaze_type == "ADAM":
            if self.atantion_optim:
                self.ADAM_optim_atantion(lerning_rate)
                return
            self.Adam_optim_conv(lerning_rate) if self.conv_optim == True else self.ADAM_optim(lerning_rate)

    def SVM_optim_atantion(self, lerning_rate):
        for key in self.Wight[self.on_this]:
         self.Wight[self.on_this][key] -= self.d_atantion_W[key] * lerning_rate

    def ADAM_optim_atantion(self, lerning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for key in self.Wight[self.on_this]:
            m_key = np.zeros_like(self.Wight[self.on_this][key])
            v_key = np.zeros_like(self.Wight[self.on_this][key])

            m_key = beta1 * m_key + (1-beta1) * self.d_atantion_W[key]
            v_key = beta2 * v_key + (1-beta2) * (self.d_atantion_W[key]**2)

            hat_m = m_key / (1-beta1 ** self.time)
            hat_v = v_key / (1-beta2 ** self.time)

            self.Wight[self.on_this][key] -= lerning_rate * hat_m / (np.sqrt(hat_v) + epsilon).tolist()


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
            self.color.append(1)
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
        scatter.set_array(np.array(self.color))

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
        scatter = ax.scatter([], [], c=self.color, cmap='viridis')
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

                #grided_out = self.grid_spliter(x_batch, 3, 3) # need mostly for object detection
        
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

            val_test = np.array(val_test)
            if self.is_grid:
                val_test = self.grid_spliter(val_test,3,3)
            if self.is_grid and  not self.is_word_data:
                val_test = val_test[None,:,:,:,:]

            self.farword(val_test)
            loss.append(self.categorical_cross_entropy(val_test_y, self.Output[-1]))

            self.show_model_prog(loss, scatter, epoch, line, ax)
            self.val_in_chat = False
        
        plt.ioff()
        plt.show()

model = NN(data_set_photo_num_ob)
model.Creat()
model.fit(30, 7, 'ADAM')