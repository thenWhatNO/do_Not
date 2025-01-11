import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

test = np.random.randn(2,4,4,6)

for a in range(0, 4, 2):
    for s in range(0, 4, 2):
        test[:, a:a+2, s:s+2, :] = test[:, a:a+2, s:s+2, :] * 0


teta = test.tolist()
print(teta)