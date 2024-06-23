import numpy as np
import matplotlib.pylab as plt

colors_plt = ['red', 'green']

data = {
    0 : [[1,2], [2,3], [2,4]],
    1 : [[3,2], [3,1], [5,3]]
}

data1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

np.random.seed(0)
data_x = 2 * np.random.rand(100, 1)
data_y = 4 + 3 * data_x + np.random.randn(100,1)

#plt.scatter(data_x, data_y)

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

def sigmoid(x):
    beta = 1.0
    return 1 / (1 + np.exp(-x))

loss_plot = []
i_plot = []


for i in range(300):
    R_label = np.random.randint(0,2)
    R_value = np.random.randint(0,3)

    out1 = np.dot(data[R_label][R_value], hidden.T)
    out2 = np.dot(out1, hidden2.T)
    out3 = np.dot(out2, output)

    loss = np.abs(R_label-out3)
    gradient3 = np.dot(loss, out3)
    gradient2 = np.dot(loss, hidden2)
    gradient1 = np.dot(loss, hidden)

    output -= 0.00001 * gradient3 * out2
    hidden2 -= 0.00001 * gradient2 * out1
    hidden -= 0.00001 * gradient1 * data[R_label][R_value]
    print(f" loss1 {loss}, label {R_label}, output: {out3}")

    loss_plot.append(loss)
    i_plot.append(i)
    
    plt.scatter(i, out3, color=colors_plt[R_label])

    plt.scatter(out3, out3, color='green')

    
plt.plot(i_plot, loss_plot, color='orange')
plt.show()