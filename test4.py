import numpy as np

data = [0,0,0,1]
target = 4

W1 = np.random.randn(4,4)
W2 = np.random.randn(4)

bias1 = np.random.randn(4)
bias2 = np.random.randn(1)

def relu(x):
    return np.maximum(0, x)

def relu_der(x):
    return np.where(x > 0, 1, 0)

learmimg_rate = 0.1


print(f"W1 {W1}")
print(f"W2 {W2}")
for i in range(10):
    x = relu((np.dot(data, W1)))
    print(f"x = {x}")
    y = relu((np.dot(x, W2)))
    print(f"y = {y}")

    print(f"x : {x} \n\n y : {y}")

    e = 0.5 * (y - target)**2
    print(F" the eror {e}")

    delta2 = (y - target)
    delta1 = np.dot(delta2, W2.T) * relu_der(x)
    print(f"delta 2 : {delta2} \n delta1 : {delta1}")

    W2 -= learmimg_rate * delta2 * x
    W1 -= learmimg_rate * np.outer(data, delta1)
    print(f"w1 : {W1} \n W2: {W2}") 