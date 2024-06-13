import torch
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

