import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from test17 import NN

data_path = "plenet_data.csv"

df = pd.read_csv(data_path)
images= df['image']
labels = df['targ']

one_imag = Image.open('data' + images[0])

img_array_for_show = []
img_array = []

for i in images:
    img = Image.open('data'+ i).convert('L')
    img_resize = img.resize((120,120))
    img_2_array = np.array(img_resize)
    img_clear = np.where(img_2_array > 50, 0 ,100)
    img_one_shot = img_clear.reshape(1, -1)

    img_array.append(img_one_shot[0])
    img_array_for_show.append(img_clear)


def one_how(labels, num_class):
    return np.eye(num_class)[labels.astype(int)]

one_label = one_how(labels, 2)

print(np.shape(img_array))
print(one_label.shape)

data_set = [[img_array], one_label]

class smallnn(NN):
    def __init__(self, data):
        super().__init__(data)

        self.layers = [
            (self.relu, 120*120, 250),
            (self.relu, 250, 250),
            (self.relu, 250, 100),
            (self.relu, 100, 2)
        ]

    def test(self, point):
        self.batch_labels = []
        self.biuld_output_size(4)
        self.farword(point)

        print(self.stack_opiration[-1])
    

model = smallnn(data_set)
model.CreatLayers()
model.fit(50, 25)


model.test(img_array[0])
print(one_label[0])
plt.imshow(img_array_for_show[0])
plt.show()