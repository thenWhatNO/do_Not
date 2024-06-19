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
    plt.scatter(x, y, color=colors[i])

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
#plt.show()

class NN:
    def __init__(self, input, output):
        self.w1 = np.array(np.random.randn(3, input))
        self.w2 = np.array(np.random.randn(3))
        
    
    def farward(self, datain):
        self.datas = np.array(datain)
        self.out1 = np.dot(datain, self.w1.T)
        self.out2 = np.dot(self.out1, self.w2)      
        return self.out2
    
    def optim(self, target, lr=0.01):
        loss = target - self.out2
        print(loss)
        grad2 = np.dot(loss, self.out1)
        print(grad2)
        loss2 = np.mean(grad2-self.out1)
        print(loss2)
        grad1 = np.dot(loss2, self.datas)
        print(grad1)

        self.w2 -= lr * grad2 * self.out1
        self.w1 -= lr * grad1 * self.datas


nn = NN(2, 1)
for i in range(5):
    R = np.random.randint(0,2)
    r = np.random.randint(0,11)

    nn.farward(np.array(data[R][r]))
    #print(f"target : {R} output : {nn.out2} loss : {r - nn.out2}")
    nn.optim(R)