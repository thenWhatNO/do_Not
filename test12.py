import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
data_x = 2 * np.random.rand(50, 1)
data_y = 4 + 3 * data_x + np.random.randn(50,1)

print(f"x : {data_x[:,0]}, y : {data_y[:,0]}")

W = 0
b = 0

x = np.arange(-10,10)

def grad(x, w1, w0, y):
    y_pred = w1 * x[:,0] + w0
    return np.array([2/len(x)*np.sum((y[:,0] - y_pred)) * (-1),
                     2/len(x)*np.sum((y[:,0] - y_pred) * (-x[:,0]))])

for i in range(150):
    w1 = W
    w2 = b

    b = w2 - 0.1 * grad(data_x, w1, w2, data_y)[0]
    W = w1 - 0.1 * grad(data_x, w1, w2,data_y)[1]
print(f"W and b {W, b}")

model = W * x + b
real_model = 3 * x + 100
plt.grid()
plt.plot(x, real_model, '--g')
plt.plot(x, model, 'r')
plt.scatter(data_x, data_y)
plt.show()