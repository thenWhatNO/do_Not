import numpy as np
import matplotlib.pyplot as plt

data = {
    0 : [[5,8],[10,6],[8,18],[20,12],[18,15],[13,5],[14,17],[20,13],[16,17],[19,7],[5,14]],
    1 : [[10,10],[11,10],[10,12],[12,12],[13,11],[12,13],[11,14],[11,12],[11,11],[10,11],[9,13]]
}

colors = ['red', 'blue']

for i, points in data.items():
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.scatter(x, y , color=colors[i])

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
#plt.show()

class NN:
    def __init__(self, input, output):
        self.layars = [
            0.10 * np.random.randn(input, 2),
            0.10 * np.random.randn(2,output)
        ]
    
    def farward(self, datain):
        self.datas = datain
        self.out1 = np.dot(datain, self.layars[0])
        self.out2 = np.dot(self.out1, self.layars[1])
        return self.out2
    
    def optim(self, target, lr=0.1):
        loss = target - self.out2
        grad2 = loss * self.out1

        grad1 = grad2 * self.datas

        print(f"shapes grad2: {np.shape(self.layars[1])} grad1: {np.shape([grad2 * lr * self.out1])}")

        self.layars[1] += [grad2 * lr * self.out1]
        self.layars[0] += grad1 * lr * self.datas

nn = NN(2, 1)
hat_y = nn.farward(data[0][0])
print(hat_y)
nn.optim(0)

hat_y = nn.farward(data[0][0])
print(hat_y)