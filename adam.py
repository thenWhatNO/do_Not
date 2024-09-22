import numpy as np

class AdamOptimizer:
    def __init__(self, lerning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lerning_rate = lerning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def initalize(self, params_shape):
        self.m = np.zeros(params_shape)
        self.v = np.zeros(params_shape)

    def updata(self, params, grads):
        if self.m is None or self.v is None:
            self.initalize(len(params))

        self.t += 1

        self.m = self.beta1 * self.m + (1-self.beta1) * grads
        self.v = self.beta2 * self.v + (1-self.beta2) * (grads**2)

        hat_m = self.m / (1-self.beta1 ** self.t)
        hat_v = self.v / (1-self.beta2 ** self.t)

        params_updata = self.lerning_rate * hat_m / (np.sqrt(hat_v)+self.epsilon)
        params -= params_updata
        return params
    
input = [[1,0,0],[1,1,0],[1,1,1]]
W = [0.-3,0.1,0.3]
bias = 1

targ = [0.1, 0.5, 1]

optim = AdamOptimizer()
t = 0

beta1 = 0.9
beta2 = 0.999
epsilon=1e-8

lerning_rate = 0.01

m = np.zeros(3)
v = np.zeros(3)

for i in range(200):
    t += 1

    R = np.random.randint(0,3)

    y = np.dot(input[R], W)
    
    loss = targ[R] - y
    
    gradient = []
    for i in range(3):
        o = loss * np.abs(input[R][i])
        gradient.append(o)

    m = beta1 * m + (1-beta1) * np.array(gradient)
    v = beta2 * v + (1-beta2) * (np.array(gradient)**2)

    hat_m = m / (1-beta1 ** t)
    hat_v = v / (1-beta2 ** t)

    W += lerning_rate * hat_m / (np.sqrt(hat_v)+epsilon)
  
   
    print(f"loss : {loss}, target : {targ[R]}, output N : {y}")