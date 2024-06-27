import numpy as np
import matplotlib.pylab as plt

colors_plt = ['red', 'green']

np.random.seed(0)
data_x = 2 * np.random.rand(50, 1)
data_y = 4 + 3 * data_x + np.random.randn(50,1)

#plt.scatter(data_x, data_y)

hidden = np.array([[0.5],[0.5]])
output = np.array([0.1, 0.1])

loss_plot = []
i_plot = []

for i in range(300):
    R_label = np.random.randint(0,len(data_x))

    out1 = np.dot(data_x[R_label], hidden.T)
    out3 = np.dot(out1, output)

    loss = np.abs(data_y[R_label][0]-out3)
    gradient3 = np.dot(loss, out3)
    gradient1 = np.dot(loss, hidden)

    output -= 0.01 * gradient3
    hidden -= 0.01 * gradient1
    print(f" loss1 {[loss]}, label {data_y[R_label]}, output: {[out3]}")

    loss_plot.append(loss)
    i_plot.append(i)
    
    
plt.plot(i_plot, loss_plot, color='orange')
plt.show()