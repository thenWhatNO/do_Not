import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

color = True

data_path = "data_2/latin_label.csv"

df = pd.read_csv(data_path)
images= df['image']
labels = df['targ']

one_imag = Image.open('data_2/latin_data_jpg/' + images[0])

img_array_for_show = []
img_array = []

for i in images:
    print('data_2/latin_data_jpg/'+ i)
    img = Image.open('data_2/latin_data_jpg/'+ i)
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

one_label = one_how(labels, 26)

data_set_photo = [img_array_for_show, one_label]