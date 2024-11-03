import numpy as np
import random

from test17 import NN

ara = np.array([[1,2,3],
                [4,0,2],
                [0,0,0]])

opa = np.where(3 == ara)

opa_list = list(zip(opa[0], opa[1]))


print(opa_list[0][0], opa_list[0][1])