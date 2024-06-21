import numpy as np

input = np.array([1.0,5.0])
hidden = np.array(
    [[0.5,0.6],
     [0.7,0.8],
     [0.1,0.2]])
hidden2 = np.array([
    [0.2,0.4,0.5],
    [0.6,0.8,0.9]
])
output = np.array([0.1, 0.2])

def softmax(x, beta=1.0):
    np.exp(beta * x) / np.sum(np.exp(beta * x))

sofmax_put = softmax(input)
print(f"softmax {sofmax_put}")

for i in range(1):
    out1 = np.dot(input, hidden.T)
    out2 = np.dot(out1, hidden2.T)
    out3 = np.dot(out2, output)

    loss = 2-out3
    gradient3 = np.dot(loss, out3)
    gradient2 = np.dot(hidden2, loss)
    gradient1 = np.dot(hidden, loss)

    output += 0.01 * gradient3
    hidden2 += 0.01 * gradient2
    hidden += 0.01 * gradient1 

    print(f"target 2, loss1 {loss}, output: {out3}")
