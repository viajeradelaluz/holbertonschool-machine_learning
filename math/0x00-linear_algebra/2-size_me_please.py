def matrix_shape(matrix):
    """Calculates the shape of a matrix.
    - You can assume all elements in the same dimension are of the same type/shape
    - The shape should be returned as a list of integers
    """
    shape = []
    while type(matrix) == list:
        print(f"matrix: {matrix}")
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
