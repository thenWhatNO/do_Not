x = [[1,2,3],
     [4,5,6,7],
     [7,8,9]]

y = [[[1,2,3],
     [4,5,6],
     [7,8,9]]]



class EliNum:

    def __init__(self):
        self.data = {}
        self.shape = None

def generait_data(data):
    
    first_shape = len(data)

    shape_of_array = [first_shape]
    scan_box = data.copy()
    shape_boxs = []

    shaps = []

    run = True
    while run:

        try:
            for layer in scan_box:
                [shape_boxs.append(x) for x in layer]
                shaps.append(len(layer))
        except Exception as e:
            run = False

        if all(ind != shaps[0] for ind in shaps):
            print("the shape is not the same!")
            break
        
        shape_of_array.append(shaps[0])

        scan_box = shape_boxs.copy()
        shape_boxs = []
    
    return shape_of_array

test = generait_data(y)