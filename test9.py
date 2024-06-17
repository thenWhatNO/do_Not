import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

class data:
    def __init__(self, data_dir, bach_num):
        self.banch = []
        self.data_image = data_dir
        df = pd.read_csv(self.data_image)
        images, classes = df['image'], df['targ']

        for i in range(bach_num):
            r = np.random.randint(0,len(images))
            img = Image.open('data' + images[r]).convert('L')
            rec_img = img.resize((120,120))
            img_arr = np.array(rec_img) 
            self.bin_img = np.where(img_arr > 50, 0, 100)
            self.banch.append([self.bin_img, classes[r]])

    def get_banch(self):
        return self.banch

    def show(self, imag):
        plt.imshow(imag)
        plt.axis('off')
        plt.show()

class NN:
    def __init__(self, in_numbers, out_nambers):
        self.m3 = np.zeros((50, out_nambers))
        self.v3 = np.zeros((50, out_nambers))
        self.m2 = np.zeros((100, 50))
        self.v2 = np.zeros((100, 50))
        self.m1 = np.zeros((in_numbers, 100))
        self.v1 = np.zeros((in_numbers, 100))

        self.classes_num = out_nambers
        self.layers_NN = [
            [0.10 * np.random.rand(in_numbers, 100), np.zeros((1, 100))],
            [0.10 * np.random.rand(100, 50), np.zeros((1,50))],
            [0.10 * np.random.rand(50, out_nambers), np.zeros((1, out_nambers))]
        ]

    def Farward(self,batch_get, targ):
        self.targ = targ
        self.batch = batch_get
        self.out1 = np.dot(self.batch, self.layers_NN[0][0]) + self.layers_NN[0][1]
        self.r_out1 = self.Relu(self.out1)
        self.out2 = np.dot(self.r_out1, self.layers_NN[1][0]) + self.layers_NN[1][1]
        self.r_out2 = self.Relu(self.out2)
        self.out3 = np.dot(self.r_out2, self.layers_NN[2][0]) + self.layers_NN[2][1]
        print(f"output : {self.out3}")
    
    def Loss(self):
        return (self.out3 - self.targ)

    def optim(self, lerning_r = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon=1e-8, t=1): 

        loss = (self.out3 - self.targ)
        self.m3 = beta1 * self.m3 + (1-beta1) * np.array(loss)
        self.v3 = beta2 * self.v3 + (1-beta2) * (np.array(loss)**2)
        hat_m3 = self.m3 / (1-beta1 ** t)
        hat_v3 = self.v3 / (1-beta2 ** t)
        b3 = np.sum(loss, axis=0, keepdims=True)

        da2 = np.dot(loss, self.layers_NN[2][0].T)
        dz2 = da2 * self.relu_der(self.out2)
        self.m2 = beta1 * self.m2 + (1-beta1) * np.array(dz2)
        self.v2 = beta2 * self.v2 + (1-beta2) * (np.array(dz2)**2)
        hat_m2 = self.m2 / (1-beta1 ** t)
        hat_v2 = self.v2 / (1-beta2 ** t)
        b2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.layers_NN[1][0].T)
        dz1 = da1 * self.relu_der(self.out1)
        self.m1 = beta1 * self.m1 + (1-beta1) * np.array(dz1)
        self.v1 = beta2 * self.v1 + (1-beta2) * (np.array(dz1)**2)
        hat_m1 = self.m1 / (1-beta1 ** t)
        hat_v1 = self.v1 / (1-beta2 ** t)
        b1 = np.sum(dz1, axis=0, keepdims=True)


        self.layers_NN[2][0] -= lerning_r * hat_m3 / (np.sqrt(hat_v3)+epsilon)
        self.layers_NN[2][1] -= lerning_r * b3
        self.layers_NN[1][0] -= lerning_r * hat_m2 / (np.sqrt(hat_v2)+epsilon)
        self.layers_NN[1][1] -= lerning_r * b2  
        self.layers_NN[0][0] -= lerning_r * hat_m1 / (np.sqrt(hat_v1)+epsilon)
        self.layers_NN[0][1] -= lerning_r * b1

    #sub layers
    def Relu(self, x):
        return np.maximum(0, x)
    
    def relu_der(self, x):
        return np.where(x > 0,1,0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        

images = data("plenet_data.csv", 100)
banches = images.get_banch()
nn = NN(120*120, 1)

for i in range(100):
    img = banches[i][0].reshape(1, -1)
    imgs = banches[i][0]
    target = banches[i][1]
    print(f"target {target} / ")
    nn.Farward(img, target)
    print(f"loss {nn.Loss()}")
    nn.optim(t=i+1)