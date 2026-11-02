import numpy as np

def make_matrices(n: int, size: int):
    matrix_list = []
    for i in range(n):
        matrix = np.random.rand(size, size)
        matrix_list.append(matrix)

    return matrix_list

def contract_matrices(matrix_list: list):
    if not matrix_list:
        raise ValueError("matrix_list cannot be empty")

    # Make a copy so the original list is not modified
    matrix_list = matrix_list.copy()

    while len(matrix_list) > 1:
        m1 = matrix_list.pop(0)
        m2 = matrix_list.pop(0)
        matrix_list.insert(0, np.matmul(m1, m2))

    return matrix_list[0]

if __name__ == "__main__":
    matrix_list = make_matrices(10, 5)

    for matrix in matrix_list:
        print(matrix)

    result = contract_matrices(matrix_list)
    print("Resultant Matrix: \n", result)
