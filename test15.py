import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

data_path = "plenet_data.csv"

df = pd.read_csv(data_path)
images= df['image']

one_imag = Image.open('data' + images[0])

img_array = []
for i in images:
    img = Image.open('data'+ i).convert('L')
    img_resize = img.resize((120,120))
    img_2_array = np.array(img_resize)
    img_clear = np.where(img_2_array > 50, 0 ,100)
    img_one_shot = img_clear.reshape(1, -1)

    img_array.append(img_clear)

plt.imshow(one_imag)
plt.show()

plt.imshow(img_array[0])
plt.show()