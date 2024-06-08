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
            bin_img = np.where(img_arr > 50, 0, 100)
            self.banch.append([bin_img, classes[r]])
    
    def show(self):
        plt.imshow(self.bin_img)
        plt.axis('off')
        plt.show()

class NN:
    def __init__(self, in_numbers, out_nambers, inputdata):
        self.inputdata = inputdata
        self.layers_NN = [
            [0.10 * np.random.rand(in_numbers, 100), np.zeros((1, 100))],
            [0.10 * np.random.rand(100, 50), np.zeros((1,50))],
            [0.10 * np.random.rand(50, out_nambers), np.zeros((1, out_nambers))]
        ]

    def Farward(self, nerual_output):
        out1 = np.dot(self.layers_NN[0], self.layers_NN[1]) * self.layers_NN
    
    def Loss(self):
        pass

    def optim(self, gradient):
        pass

    #sub layers
    def Relu(self, x):
        return np.maximum(0, x)

images = data("plenet_data.csv", 100)
print(images.banch[0])