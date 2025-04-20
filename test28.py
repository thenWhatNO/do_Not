x = [[1,2,3],
     [4,5,6,7],
     [7,8,9]]

y = [[1,1,1],
     [2,2,2],
     [3,3,3]]

show_len = len(x)


def test_func(data):
    shape = []
    while isinstance(data, list):
        shape.append(len(data))
        data = data[0] if len(data) > 0 else []
    return shape

shapy = test_func(x)


class Matrix:
    def __init__(self, data):
        self.data = data

    class Vector:
        def __init__(self, data):
            self.vectors = self.split_data_to_vectors(data)
            self.shape = self.shape_detect(data)
            self.type = self.type_detect(data)

        def shape_detect(data):
            pass

        def type_detect(data):
            pass

        def split_data_to_vectors(data):
            shape = []

            start_shape = len(data)
            shape.append(start_shape)



        
        def recursiv_split(data):
            shapes = []
            for i in data:
                if len(i) not in shapes:
                    shapes.append(len(i))
            if len(shapes) > 1:
                raise ValueError(f"the shape is ended as {shapes}, what is not functioanal for matrix useg")
            else:
                return data[0]
    
    def __matmul__(self, other):

        row_a = len(self.data)
        col_a = len(self.data[0])
        row_b = len(other.data)
        col_b = len(other.data[0])

        output = [[0 for _ in range(col_b)] for _ in range(row_a)]

        for i in range(row_a):
            for j in range(col_b):
                for k in range(col_a):
                    output[i][j] += self.data[i][k] * other.data[k][j]

