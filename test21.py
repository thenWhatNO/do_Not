import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


input_image = [[[1,1],[1,1],[1,1],[1,1]],
               [[1,1],[1,1],[1,1],[1,1]],
               [[1,1],[1,1],[1,1],[1,1]],
               [[1,1],[1,1],[1,1],[1,1]],]


# // woek on a multy batch conv algoritem look in chat GPt
print(np.shape(input_image))

if 3 <= np.shape(input_image):
    cannals_num = len(input_image[2])

print(cannals_num)