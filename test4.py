import numpy as np

data = [0,0,0,1]
target = 4

W1 = np.random.randn(4,4)
W2 = np.random.randn(4)

bias1 = np.random.randn(4)
bias2 = np.random.randn(1)

def relu(x):
    return np.maximum(np.abs(0.1*x), x)
def relu_der(x):
    if x > 0:
        return 1
    elif x <= 0:
        return np.abs(0.1*x)

print(relu_der(0))

print(f"W1 {W1}")
print(f"W2 {W2}")
for i in range(50):
    x = relu((np.dot(data, W1)))
    y = relu((np.dot(x, W2)))

    print(f"x : {x} \n\n y : {y}")

    e = np.abs(y - target)
    print(e)

    delta1 = e*relu_der(y)
    print(delta1)

    learmimg_rate = 0.001

    W2 = W2 - learmimg_rate * delta1 * y
    print(W2)

    delta2 = W2 * delta1 * [relu_der(xx) for xx in x]
    print(delta2)

    W1 = W1 - learmimg_rate * delta2 * x
    print(W1)
