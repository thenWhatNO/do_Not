import math
import numpy as np

inputs = [[4.8, 1.21, 2.385],
          [8.9, -1.81, 0.2],
          [1.41, 1.051, 0.026]]

exp_val = np.exp(input)
norm_val = exp_val / np.sum(inputs, axis=1, keepdims=True)

print(exp_val)