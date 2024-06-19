import numpy as np

input = np.array([1,5])
hidden = np.array(
    [[0.5,0.6],
     [0.7,0.8],
     [0.1,0.2]])
output = np.array([0.1, 0.2, 0.3])

for i in range(100):
    out1 = np.dot(input, hidden.T)
    out2 = np.dot(out1, output)

    loss = 2-out2
    gradient2 = np.dot(loss,out1)
    gradient1 = np.dot(hidden, loss)

    output += 0.01 * gradient2 
    hidden += 0.01 * gradient1 

    print(f"target 2, loss1 {2-out2}, output: {out2}")
