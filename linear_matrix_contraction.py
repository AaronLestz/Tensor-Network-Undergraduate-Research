import numpy as np


def make_matrices(n: int, size: int):
    matrix_list = []
    for i in range(n):
        matrix = np.random.rand(size, size)
        matrix_list.append(matrix)

    return matrix_list


def contract_matrices(matrix_list: list):
    while (len(matrix_list) > 1):
        m1 = matrix_list.pop()
        m2 = matrix_list.pop()
        matrix_list.append(np.matmul(m1, m2))

    return matrix_list[0]


if __name__ == "__main__":
    matrix_list = make_matrices(10, 5)
    for matrix in matrix_list:
        print(matrix)
    result = contract_matrices(matrix_list)
    print("Resultant Matrix: \n", result)

    matrix_1 = np.eye(2,2)
    matrix_2 = np.eye(2,2)
    print(matrix_1)
    print(matrix_2)
print(contract_matrices([matrix_1, matrix_2]))