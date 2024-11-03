import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_apa = "data/O/5a0d5b5b66079.jpg"

img = np.array(Image.open(image_apa).convert('L').resize((120,120)))

# plt.imshow(img)
# plt.show()

def conv2d(input_image, kernel, stride=1, padding=0):
    if padding > 0:
        input_image = np.pad(input_image, ((padding, padding), (padding, padding)), mode='constant')

    input_height, input_width = input_image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width))

    for y in range(0, output_height):
        for x in range(0, output_width):
            region = input_image[y*stride:y*stride+kernel_height, x*stride:x*stride+kernel_width]
            opa = region * kernel
            output[y, x] = np.sum(opa)

    return output

def conv2D_drev(input_image, kernel, gradient, stride=1, padding=0, lerning_rate = 0.01):
    if padding > 0:
        input_image = np.pad(input_image, ((padding, padding), (padding, padding)), mode='constant')

    kernel_height, kernel_width = kernel.shape
    gradient_height, gradient_width = gradient.shape

    filter_gradient = np.zeros((kernel_height, kernel_width))

    for y in range(0, gradient_height):
        for x in range(0, gradient_width):
            region = input_image[y:y+kernel_height, x:x+kernel_width]
            filter_gradient += region * gradient[y,x]

    new_kernel = kernel - lerning_rate * filter_gradient
    
    return new_kernel

def poolingMax(input_image, steps=2):

    input_height, input_width = input_image.shape

    output_height = (input_height - steps) // steps + 1
    output_width = (input_width - steps) // steps + 1

    output_image = np.zeros((output_height, output_width))

    for y in range(0, output_height):
        for x in range(0, output_width):
            region = input_image[y*steps:y*steps+steps, x*steps:x*steps+steps]
            opa = np.max(region)
            output_image[y, x] = opa

    return output_image

def poolimgMax_drev(input_image, steps=2):

    input_height, input_width = input_image.shape

    find = 1

    useg_height = (input_height - steps) // steps + 1
    useg_width = (input_width - steps) // steps + 1

    output_image = np.zeros((input_height, input_width))

    for y in range(0, useg_height):
        for x in range(0, useg_width):
            region = input_image[y*steps:y*steps+steps, x*steps:x*steps+steps]
            maxy = np.max(region)
            wer = np.where(region == maxy)
            wer_list = list(zip(wer[0], wer[1]))
            output_image[y*steps+wer_list[0][0], x*steps+wer_list[0][1]] = find
            find += 1
    
    return output_image

def Flatting(img):
    return np.array(img).flatten()

def relu(x):
    return np.where(x >= 0, x, 0)

input_image = np.array([[1, 2, 3, 0],
                        [0, 1, 2, 3],
                        [3, 0, 1, 1],
                        [2, 1, 0, 2]])

kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

gradient = np.array([[2,2],
                     [2,2]])

output = conv2d(input_image, kernel, stride=1, padding=1)
print(output)

hopa = relu(output)
print(hopa)

ara = poolingMax(hopa)
print(ara)

flating = Flatting(ara)
print(flating)