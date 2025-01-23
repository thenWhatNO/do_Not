import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

test = np.random.randn(4,4,6)
other = test.reshape(4*4,6)

for i in range(2):
    for ii in range(0,4, 2):
        other[:,i:ii] = i+i
        test[:,:,i:ii] = i+i

opa = other.tolist()
opo = test.tolist()

help = other.reshape(4,4,6).tolist()
print(1)