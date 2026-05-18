import numpy as np

# Defining proper direction order as (left, up right, down)
def make_n_by_three_network(n: int):
    # 3n-3 - 3n-2 - 3n-1
    # ...
    #  0   -   1  -   2
    tensor_list = []
    for j in range(3):
        if(j == 0 or j == 2):
            # edge matrix
            mat = np.random.rand(3,3)
        else:
            # middle matrix
            mat = np.random.rand(3,3,3)
        tensor_list.append(mat)
    for i in range(n - 2):
        for j in range(3):
            if(j == 0 or j == 2):
                mat = np.random.rand(3,3,3)
            else:
                mat = np.random.rand(3,3,3,3)
            tensor_list.append(mat)
    for j in range(3):
        if(j == 0 or j == 2):
            mat = np.random.rand(3,3)
        else:
            mat = np.random.rand(3,3,3)
        tensor_list.append(mat)
    return tensor_list

"""
def svd_compression_1xn(tn_1xn: list[np.ndarray]):
        intermediates = []

        # Prior to having an intermediate E
        tensor = tn_1xn[0]
        intermediate = np.einsum("lur, luR -> rR", tensor, tensor.conj())
        intermediates.append(intermediate)

        for i in range(1, len(tn_1xn)): # Now have an intermediate E
            tensor = tn_1xn[i]
            prev_intermediate = intermediates[-1]
            intermediate = np.einsum("lL, lur, LuR -> rR", \
                                     prev_intermediate, tensor, tensor.conj())
            intermediates.append(intermediate)

        # TODO:: All steps past this contraction (Density Matrix first)

        return "INCOMPLETE"
"""

def generate_left_canonical_form(mps: list[np.ndarray]):
    """
    In place algorithm to generate left canonical form
    of the given matrix product states
    """
    for i in range(len(mps) - 1): # Leave last column
        tensor = mps[i]
        l, p, r = tensor.shape
        tensor = tensor.reshape(l * p, r)

        U, S, Vdag = np.linalg.svd(tensor, full_matrices=False)
        U = U.reshape(l, p, S.shape[0])
        S_Vdag = np.matmul(np.diag(S), Vdag)
        mps[i] = U
        mps[i + 1] = np.einsum("le, eur -> lur", S_Vdag, mps[i + 1])
def mps_svd_contraction(mps: list[np.ndarray], dim: int = -1,
                            lose_dim: int = -1, comp_percent: float = 1):
    """
    In place algorithm to compress a given matrix product state (mps)
    Compresses based on the 3 other arguments provided:
        dim - sets dimension of connecting indices to this argument
        lose_dim - removes lose_dim number of dimensions from the indices
        comp_percent - retain (comp_percent)% of the dimensions, rounded down
    If a compression argument is not given, does not compress in its manner
    """
    generate_left_canonical_form(mps) # In place
    for i in reversed(range(1, len(mps))):
        tensor = mps[i]
        l, p, r = tensor.shape
        tensor = tensor.reshape(l, p * r)

        U, S, Vdag = np.linalg.svd(tensor, full_matrices=False)
        if(dim != -1):
            curr_dim = min(len(S), dim)
            U = U[:, :curr_dim]
            S = S[:curr_dim]
            Vdag = Vdag[:curr_dim, :]
        if(lose_dim != -1 and len(S) > lose_dim):
            curr_dim = len(S) - lose_dim
            U = U[:, :curr_dim]
            S = S[:curr_dim]
            Vdag = Vdag[:curr_dim, :]
        if(comp_percent > 0 and comp_percent < 1):
            curr_dim = int(len(S) * comp_percent)
            U = U[:, :curr_dim]
            S = S[:curr_dim]
            Vdag = Vdag[:curr_dim, :]

        Vdag = Vdag.reshape(S.shape[0], p, r)
        U_S = np.matmul(U, np.diag(S))
        mps[i] = Vdag
        mps[i - 1] = np.einsum("lue, er -> lur", mps[i - 1], U_S)

def contract_n_by_three_network(tensor_list: list[np.ndarray], compress: bool = False,
                                dim: int = -1, lose_dim: int = -1, comp_percent: float = 1):
    def modular_2x_1x3_contraction(bottom_1x3: list, top_1x3: list):
        # Contract each pair of tensors in bottom_1x3 and top_1x3
        def set_proper_tensor_orders(tensor: np.ndarray, index: int, is_top: bool):
            shape = tensor.shape
            order = len(shape)
            if(is_top):
                if(order == 3):
                    if(index == 0): # left
                        return tensor.reshape(1, shape[0], shape[1], shape[2])
                    else: # right
                        return tensor.reshape(shape[0], shape[1], 1, shape[2])
                else: # order 4
                    return tensor
            else: # bottom
                if(order == 2):
                    if(index == 0): # left
                        return tensor.reshape(1, shape[0], shape[1])
                    else: # right
                        return tensor.reshape(shape[0], shape[1], 1)
                else: # order 3
                    return tensor
                
        contracted_1x3 = []
        for i in range(3):
            bottom = set_proper_tensor_orders(bottom_1x3[i], i, False)
            top = set_proper_tensor_orders(top_1x3[i], i, True)
            false_contracted: np.ndarray = np.einsum("ler, LURe -> lLUrR", bottom, top)
            contracted_left = false_contracted.shape[0] * false_contracted.shape[1]
            contracted_right = false_contracted.shape[3] * false_contracted.shape[4]
            contracted = false_contracted.reshape(contracted_left, false_contracted.shape[2], contracted_right)
            contracted_1x3.append(contracted)
        if(compress):
            mps_svd_contraction(contracted_1x3, dim, lose_dim, comp_percent)
        return contracted_1x3
    
    def final_contraction(bottom_1x3: list, top_1x3: list):
        contracted_1x3 = []
        for i in range(3): # contract into final 1x3
            bottom = bottom_1x3[i]
            top = top_1x3[i]
            shape = top.shape
            if(i == 0): # left
                top = top.reshape(1, shape[0], shape[1])
            elif(i == 2): # right
                top = top.reshape(shape[0], 1, shape[1])
            
            false_contracted = np.einsum("ler, LRe -> lLrR", bottom, top)
            contracted_left = false_contracted.shape[0] * false_contracted.shape[1]
            contracted_right = false_contracted.shape[2] * false_contracted.shape[3]
            contracted = false_contracted.reshape(contracted_left, contracted_right)
            contracted_1x3.append(contracted)
        
        left_middle_contracted = np.matmul(contracted_1x3[0], contracted_1x3[1])
        final_contracted = np.matmul(left_middle_contracted, contracted_1x3[2])
        return final_contracted
    
    bottom_1x3 = tensor_list[0:3]
    for i in range(1, (len(tensor_list) // 3) - 1):
        top_1x3 = tensor_list[3*i:3*i+3]
        bottom_1x3 = modular_2x_1x3_contraction(bottom_1x3, top_1x3)
    top_1x3 = tensor_list[-3:]
    return np.squeeze(final_contraction(bottom_1x3, top_1x3))

if __name__ == "__main__":
    n: int = int(input("Number of Rows of Tensor List "))
    tensor_list = make_n_by_three_network(n)
    for tensor in tensor_list:
        print(tensor)
    if(n <= 20):
        result_uncompressed = contract_n_by_three_network(tensor_list)
        print("Result of uncompressed contraction: \n", result_uncompressed)
    result_compressed = contract_n_by_three_network(tensor_list, compress=True, dim=2)
    result_compressed_2 = contract_n_by_three_network(tensor_list, compress=True, lose_dim=1)
    result_compressed_3 = contract_n_by_three_network(tensor_list, compress=True, comp_percent=.75)
    print("Result of compressed contraction: \n", result_compressed)
    print("Result of second compressed contraction: \n", result_compressed_2)
    print("Result of compressed contraction: \n", result_compressed_3)