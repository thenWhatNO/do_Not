import numpy as np

data = [2,4,6,8,10,12,14,16,18,20]

targets = [22]

W1 = np.random.randn(10)
W2 = np.random.randn(1)

def relu(x):
    return np.maximum(0,x)

def der_relu(x):
    return np.where(x>0,1,0)

lerning_R = 0.1

for i in range(30):
    rend = np.random.randint(0,4)
    O1 = relu(np.dot(data[rend], W1.T))
    O2 = relu(np.dot(O1, W2))
    
    loss = 0.5 * (O2 - targets[rend])**2

    delta2 = (O2 - targets[rend])
    delta1 = np.dot(delta2,W2.T) * der_relu(O1) 

    W2 -= lerning_R * delta2 * O1
    W1 -= lerning_R * np.outer(data[rend], delta1)
    print(f"loss : {loss}   target : {targets[rend]}   Y : {O2}")

def test(x):
    O1 = relu(np.dot(data[rend], W1.T))
    O2 = relu(np.dot(O1, W2))
    return O2
print(f"test  data{data[0]}, the output : {test(data[0])}")