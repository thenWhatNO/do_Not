import numpy as np

x = np.random.randint(0, 3, (1, 3, 3, 20))

b = np.transpose(x, (1,2,0,3))

x_list = x.tolist()
b_list = b.tolist()

pr = np.size(b[:,1,0])
px = np.size(x[:,:,1,1,:])

print(b[:,0,0])

print(x[-1,-1])