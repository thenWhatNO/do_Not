x = [[1,2,3],
     [4,5,6],
     [7,8,9]]

y = [[1,2,3],
     [4,5,6],
     [7,8,9]]

z = [[1,2],
     [3,4],
     [5,6],
     [7,8],
     [9,10]]

import numpy as np

def generait_shape(data): 
    first_shape = len(data)

    shape_of_array = [first_shape]
    scan_box = data.copy()
    shape_boxs = []

    true_value = None

    shaps = []

    run = True
    while run:

        try:
            for layer in scan_box:
                [shape_boxs.append(x) for x in layer]
                shaps.append(len(layer))
        except Exception as e:
            run = False
            true_value = scan_box.copy()
            break

        if all(ind != shaps[0] for ind in shaps):
            print("the shape is not the same!")
            break
        
        shape_of_array.append(shaps[0])

        scan_box = shape_boxs.copy()
        shape_boxs = []

    return shape_of_array, true_value

def revers(data):
    try:
        data_len = len(data)
        new_array = []

        for i in range(-1, -(data_len+1), -1):
            new_array.append(data[i])

        return new_array
    except Exception as e:
        print("can't revers!") 

def transForm(data, shape):
    revers_shape = revers(shape)
    revers_shape.pop()
    copy_array = []

    _, true_value = generait_shape(data)
    working_array = true_value.copy()

    ind = 0

    try:
        for times in revers_shape:
            while len(working_array) != 0:
                array = working_array[0:times]
                copy_array.append(array)
                [working_array.pop(0) for ind in range(0,(times))]
            working_array = copy_array.copy()
            copy_array = []
    except Exception as e:
        print("can't transfomr the data!") 

    return working_array

def generate_indexs(data):
    shape, true_value = generait_shape(data)
    index_list = []
    other_list = []

    len_of = len(true_value)

    [index_list.append([i]) for i in range(shape[0])]
    
    for index, time in enumerate(shape):
        if index + 1 >= len(shape):
            break
        for ind in range(time):
            working_ely = [index_list[ind].copy() for i in range(shape[index+1])]

            [working_ely[j].append(j) for j in range(shape[index+1])]
            [other_list.append(ely) for ely in working_ely]

        index_list = other_list.copy()
        other_list = []

    return_data = []
    [return_data.append([index_list[i], true_value[i]]) for i in range(len_of)]

    return return_data  


class EliNum:

    def __init__(self, data):
        self.data = data
        self.shape, self.true_value = generait_shape(data)
        self.index_coll = generate_indexs(data)
    
    
    def __getitem__(self, ind):
        return_num = self.data.copy()
        for i in ind:
            return_num = return_num[i]
        return return_num
    
    def __setitem__(self, ind, value):
        pass

    def __len__(self, deph):
        return self.shape[deph]
    
    def __add__(self, other):
        if len(self.true_value) != len(other.true_value):
            raise print("not the same size of the array to sum!")
        
        new_array = []

        work_self_array = self.true_value
        work_other_array = other.true_value

        for ind in range(len(self.true_value)):
            sum_value = work_self_array[ind] + work_other_array[ind]
            new_array.append(sum_value)
        
        return transForm(new_array, self.shape)

        
    def __radd__(self,other):
        pass

    def __sub__(self,other):
        if len(self.true_value) != len(other.true_value):
            raise print("not the same size of the array to sum!")
        
        new_array = []

        work_self_array = self.true_value
        work_other_array = other.true_value

        for ind in range(len(self.true_value)):
            sum_value = work_self_array[ind] - work_other_array[ind]
            new_array.append(sum_value)
        return transForm(new_array, self.shape)

    def __rsub__(self,other):
        pass

    def __mul__(self,other):
        pass

    def __rmul__(self,other):
        pass

    def __truediv__(self,other):
        pass

    def __rtruediv__(self,other):
        pass

    def __floordiv__(self,other):
        pass

    def __pow__(self,other):
        pass

    def __neg__(self,other):
        pass

    def __abs__(self,other):
        pass


my_data = EliNum(y)
my_za = EliNum(x)

new_data = my_data + my_za
print(new_data)