import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

test = np.random.randn(1,4,4,6)
other = test.reshape(1,4*4,6)

for i in range(2):
    for ii in range(0,4, 2):
        other[:,i:ii] = i+i
        test[:,:,i:ii] = i+i

opa = other[0]
opo = test[0]

help = other.reshape(4,4,6).tolist()
print(1)