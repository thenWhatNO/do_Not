import numpy as np

def f(x):
    return 2/(1 + np.exp(-x)) - 1

def fd(x):
    return 0.5 * (1 + x) * (1 - x)

W1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, 0.4]])
W2 = np.array([0.2, 0.3])

def go_forward(inp):
    sum = np.dot(W1, inp)
    print(sum.shape)
    out = np.array([f(x) for x in sum])
    print(out.shape)

    sum = np.dot(W2, out)
    print(sum.shape)
    y = f(sum)
    return (y, out)

def train(epoch):
    global W1, W2
    lmb = 0.01
    N = 1
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]
        y, out = go_forward((x[0:3]))
        e = y - x[-1]
        delta = e * fd(y)
        print(delta)
        W2[0] = W2[0] - lmb * delta * out[0]
        W2[1] = W2[1] - lmb * delta * out[1]

        delta2 = W2*delta*fd(out)
        print(delta2)

        W1[0,:] = W1[0,:] - np.array(x[0:3]) * delta2[0] * lmb
        W1[1,:] = W1[1,:] - np.array(x[0:3]) * delta2[1] * lmb

epoch =[(-1,-1,-1,-1),
        (-1,-1,1,1),
        (-1,1,-1,-1),
        (-1,1,1,1),
        (1,-1,-1,-1),
        (1,-1,1,1),
        (1,1,-1,-1),
        (1,1,1,-1)]

train(epoch)

for x in epoch:
    y, out = go_forward(x[0:3])


