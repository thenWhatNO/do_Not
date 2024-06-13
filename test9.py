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

    def show(self):
        plt.imshow(self.bin_img)
        plt.axis('off')
        plt.show()

class NN:
    def __init__(self, in_numbers, out_nambers):
        self.classes_num = out_nambers
        self.layers_NN = [
            [0.10 * np.random.rand(in_numbers, 100), np.zeros((1, 100))],
            [0.10 * np.random.rand(100, in_numbers), np.zeros((1,in_numbers))],
            [0.10 * np.random.rand(in_numbers, out_nambers), np.zeros((1, out_nambers))]
        ]

    def Farward(self,batch_get, targ):
        self.targ = targ
        self.batch = batch_get
        self.out1 = np.dot(self.batch, self.layers_NN[0][0]) + self.layers_NN[0][1]
        self.r_out1 = self.Relu(self.out1)
        self.out2 = np.dot(self.r_out1, self.layers_NN[1][0]) + self.layers_NN[1][1]
        self.r_out2 = self.Relu(self.out2)
        self.out3 = np.dot(self.r_out2.T, self.layers_NN[2][0]) + self.layers_NN[2][1]
        print(self.out3)
    
    def Loss(self):
        pass

    def optim(self): 
        self.labels = []
        if self.targ == 0:
            self.labels = [0,1]
        elif self.targ == 1:
            self.labels = [1,0]

        loss = (self.out3 - self.targ) / np.size(self.targ)
        W3 = np.dot(self.r_out2.T, loss)

        da2 = np.dot(loss, self.layers_NN[2][0].T)
        dz2 = da2 * self.relu_der(self.out2)
        W2 = np.dot(self.r_out1.T, dz2)

        da1 = np.dot(dz2, self.layers_NN[1][0].T)
        dz1 = da1 * self.relu_der(self.out1)
        W1 = np.dot(self.batch, dz1)

        self.layers_NN[2][0] += 0.001 * W3
        self.layers_NN[1][0] += 0.001 * W2
        self.layers_NN[0][0] += 0.001 * W1

    #sub layers
    def Relu(self, x):
        return np.maximum(0, x)
    
    def relu_der(self, x):
        return np.where(x > 0,1,0)

    def f(self, x):
        return 2/(1 + np.exp(-x)) - 1

images = data("plenet_data.csv", 100)
banches = images.get_banch()
nn = NN(120, 2)


print(f"target {banches[0][1]}")
nn.Farward(banches[0][0], banches[0][1])
nn.optim()
print(f"target {banches[0][1]}")
nn.Farward(banches[0][0], banches[0][1])