import numpy as np
import matplotlib.pyplot as plt

x_min, x_max, y_min, y_max = -3.5, 3.5, -3.5, 3.5
step_size = 0.3

x_values = np.arange(x_min, x_max, step_size)
y_values = np.arange(y_min, y_max, step_size)
x_grid, y_grid = np.meshgrid(x_values, y_values)

x_point = x_grid.flatten()
y_point = y_grid.flatten()

plt.scatter(x_point, y_point, color="black", marker='o')
plt.grid()

n_samples_per_cluster = 50  
n_features = 2        
n_clusters = 4

means = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
std_devs = [0.5, 0.5, 0.5, 0.5]

data_x = np.zeros((n_samples_per_cluster * n_clusters, n_features))
data_y = np.zeros(n_samples_per_cluster * n_clusters)

np.random.seed(1)

for i in range(n_clusters):
    data_x[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster] = np.random.normal(
        loc=means[i], scale=std_devs[i], size=(n_samples_per_cluster, n_features))
    data_y[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster] = i

def one_how(labels, num_class):
    return np.eye(num_class)[labels.astype(int)]

data_y_one_hot = one_how(data_y, n_clusters)


print(np.array([[x_point[0]], [y_point[0]]]), data_x[i].reshape(-1, 1))
plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, cmap='viridis')
plt.show()

def relu(x):
    return np.where(x >= 0, x, 0.001*x)
def derv_relu(x):
    return np.where(x >= 0, x, 0.001)

def sigmoid(x):
    if np.any(x > 300) or np.any(x < -300):
        x = np.clip(x, -300, 300) + np.random.normal(0, 0.1, x.shape)
    return  1 / (1 + np.exp(-x))

def sigmoid_derv(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

np.random.seed(1)

W1 = np.random.rand(25, n_features)
b1 = np.random.rand(1, 25)
W2 = np.random.rand(25, 25)
b2 = np.random.rand(1, 25)
W3 = np.random.rand(n_clusters, 25)
b3 = np.random.rand(1, n_clusters)

def farword(x):
    z1 = np.dot(x.T, W1.T) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2.T) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, W3.T) + b3
    a3 = sigmoid(z3)
    return z1, a1, z2, a2, z3, a3

def backword(x, y):
    z1, a1, z2, a2, z3, a3 = farword(x)

    d_a3 = 2 * (a3 - y)
    d_z3 = d_a3 * sigmoid_derv(z3)
    d_W3 = np.dot(d_z3.T, a2)
    d_b3 = np.sum(d_z3, axis=0, keepdims=True)

    d_a2 = np.dot(d_a3, W3)
    d_z2 = d_a2 * derv_relu(z2)
    d_W2 = np.dot(d_z2.T, a1)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_a2, W2)
    d_z1 = d_a1 * derv_relu(z1)
    d_W1 = np.dot(d_z1.T, x.T)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    # print('\n X:', x, '\n y:', y ,'\n a1:', 
    #       d_a1, "\n a2:", d_a2, '\n a3:', d_a3, '\n z1:',
    #       d_z1, '\n z2:', d_z2, '\n z3:', d_z3, '\n W3:',
    #       d_W3, '\n W2:', d_W2, '\n W1', d_W1)

    return d_W1, d_b1, d_W2, d_b2, d_W3, d_b3

for ii in range(700):
    for i in range(len(data_x)):
        X = data_x[i].reshape(-1, 1)
        Y = np.array([data_y_one_hot[i]])

        d_W1 , d_b1, d_W2, d_b2, d_W3, d_b3  = backword(X, Y)

        W1 -= 0.001 * d_W1
        b1 -= 0.001 * d_b1
        W2 -= 0.001 * d_W2 
        b2 -= 0.001 * d_b2
        W3 -= 0.001 * d_W3
        b3 -= 0.001 * d_b3

plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, alpha=0.4, cmap='viridis')
prid = []
for i in range(len(x_point)):
    X = np.array([[x_point[i]], [y_point[i]]])
    #Y = data_y_one_hot[i]

    _, _, _, _, _, a3 = farword(X)

    output = np.round(a3)
    #print(f"it : {i}, pred : {output}")
    
    if output[0][0] == 1:
        prid.append('red')
    elif output[0][1] == 1:
        prid.append('blue')
    elif output[0][2] == 1:
        prid.append('green')
    elif output[0][3] == 1:
        prid.append('yellow')
    else:
        prid.append('black')

plt.scatter(x_point, y_point, c=prid, marker="x", cmap='viridis')

# prid2 = []
# for i in range(len(data_x)):
#     X = data_x[i].reshape(-1, 1)
#     #Y = data_y_one_hot[i]

#     _, _, _, _, _, a3 = farword(X)

#     output = np.round(a3)
#     #print(f"it : {i}, pred : {output}")
    
#     if output[0][0] == 1:
#         prid2.append(0)
#     elif output[0][1] == 1:
#         prid2.append(1)
#     elif output[0][2] == 1:
#         prid2.append(2)
#     elif output[0][3] == 1:
#         prid2.append(3)
#     else:
#         prid.append(4)

# plt.scatter(data_x[:, 0], data_x[:, 1], c=prid2, marker="x", cmap='viridis')

plt.show()