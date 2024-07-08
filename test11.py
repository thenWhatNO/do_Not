import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
data_x = np.array(3 * np.random.rand(50, 1))
data_y = np.array(1 + data_x**4 + np.random.rand(50, 1)**2)

#plt.scatter(data_x, data_y)

W1 = np.random.rand(10,1)
b1 = np.random.rand(10)

outW = np.random.rand(1,10)
outb = np.random.rand(1)

def relu(X):
    return np.maximum(0, X)

def derv_relu(x):
    return np.where(x > 0,1,0)

I_arr = []
YS_arr = []


for i in range(4000):

    R = np.random.randint(0, len(data_x))
    Y = data_y[R][0]
    steps = data_x[R][0]

    Z = np.dot(steps, W1[0]) + b1
    A = np.array([relu(Z)])

    ZZ = np.dot(A, outW.T) + outb
    AA = relu(ZZ)

    DA = -(2 / data_x.shape[0]) * (Y - AA)
    
    DZ = DA * derv_relu(AA)
    DW = np.dot(A.T, DZ)
    DB = np.sum(DZ, axis=0, keepdims=True)

    DA2 = np.dot(DZ, outW)
    DZ2 = DA2 * derv_relu(A)
    DW2 = np.dot(steps, DZ)
    DB2 = np.sum(DZ2, axis=0, keepdims=True)

    outW -= 0.01 * DW.T
    outb -= 0.01 * DB[0]

    W1 -= 0.01 * DW2
    b1 -= 0.01 * DB2[0]
    print(f"Y : {AA} Y_HAT : {Y} LOSS : {DA[0][0]}")
    
    I_arr.append(i)
    YS_arr.append(DA[0][0])

    #plt.scatter(steps, AA, color="red")  



for i in range(50):
    Z = np.dot(data_x[i], W1[0]) + b1
    A = np.array([relu(Z)])

    ZZ = np.dot(A, outW.T) + outb
    AA = relu(ZZ)
    plt.scatter(data_x[i], AA, color="red")        
plt.show()

print("?")

plt.scatter(I_arr, YS_arr, color="green")
plt.grid()
plt.show()