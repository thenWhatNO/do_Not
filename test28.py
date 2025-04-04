import numpy as np

x = np.random.randn(10,9,3,3,4)

b = x.transpose(*reversed(range(x.ndim)))

pr = np.size(b[:,1,0])
px = np.size(x[:,:,1,1,:])

print(b[:,0,0])

print(x[-1,-1])