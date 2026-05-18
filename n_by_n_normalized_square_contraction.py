import numpy as np
import copy

# Defining proper direction order as (left, up right, down)

def make_n_by_n_network(n: int):
    # 3n-3 - 3n-2 - 3n-1
    # ...
    #  0   -   1  -   2
    tensor_list = []
    for j in range(n):
        if(j == 0 or j == n - 1):
            # edge matrix
            mat = np.random.rand(3,3)
        else:
            # middle matrix
            mat = np.random.rand(3,3,3)
        tensor_list.append(mat)
    for i in range(n - 2):
        for j in range(n):
            if(j == 0 or j == n - 1):
                mat = np.random.rand(3,3,3)
            else:
                mat = np.random.rand(3,3,3,3)
            tensor_list.append(mat)
    for j in range(n):
        if(j == 0 or j == n - 1):
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

def compute_mps_error(exact_mps: list[np.ndarray], compressed_mps: list[np.ndarray]) -> float:
    """
    Calculates L^2 Error between exact & compressed mps
    """
    def mps_inner_product(mps1, mps2):
        E = np.einsum("lpr, lpR -> rR",
                      mps1[0].conj(),
                      mps2[0])

        for i in range(1, len(mps1)):
            E = np.einsum("lL, lpr, LpR -> rR",
                          E,
                          mps1[i].conj(),
                          mps2[i])

        return np.squeeze(E)

    norm_exact = mps_inner_product(exact_mps, exact_mps)
    norm_compressed = mps_inner_product(compressed_mps, compressed_mps)
    overlap = mps_inner_product(exact_mps, compressed_mps)

    err_sq = norm_exact + norm_compressed - 2 * np.real(overlap)
    err_sq = max(err_sq, 0.0)

    return float(np.sqrt(err_sq))

def contract_n_by_n_network(tensor_list: list[np.ndarray], n: int, compress: bool = False,
                                dim: int = -1, lose_dim: int = -1, comp_percent: float = 1):

    def modular_2x_1xn_contraction(bottom_1xn: list, top_1xn: list):
        def set_proper_tensor_orders(tensor: np.ndarray, index: int, is_top: bool):
            shape = tensor.shape
            order = len(shape)
            if(is_top):
                if(order == 3):
                    if(index == 0):
                        return tensor.reshape(1, shape[0], shape[1], shape[2])
                    else:
                        return tensor.reshape(shape[0], shape[1], 1, shape[2])
                else:
                    return tensor
            else:
                if(order == 2):
                    if(index == 0):
                        return tensor.reshape(1, shape[0], shape[1])
                    else:
                        return tensor.reshape(shape[0], shape[1], 1)
                else:
                    return tensor
                
        contracted_1xn = []
        for i, bottom_tensor in enumerate(bottom_1xn):
            bottom = set_proper_tensor_orders(bottom_tensor, i, False)
            top = set_proper_tensor_orders(top_1xn[i], i, True)

            false_contracted = np.einsum("ler, LURe -> lLUrR", bottom, top)

            contracted_left = false_contracted.shape[0] * false_contracted.shape[1]
            contracted_right = false_contracted.shape[3] * false_contracted.shape[4]

            contracted = false_contracted.reshape(contracted_left,
                                                  false_contracted.shape[2],
                                                  contracted_right)
            contracted_1xn.append(contracted)

        if compress:
            exact_copy = copy.deepcopy(contracted_1xn)
            mps_svd_contraction(contracted_1xn, dim, lose_dim, comp_percent)
            step_error = compute_mps_error(exact_copy, contracted_1xn)
            return contracted_1xn, step_error

        return contracted_1xn, 0.0


    def final_contraction(bottom_1xn: list, top_1xn: list):
        contracted_1xn = []

        for i, bottom in enumerate(bottom_1xn):
            # bottom = bottom  # From previous range(len()) version
            top = top_1xn[i]
            shape = top.shape

            if(i == 0):
                top = top.reshape(1, shape[0], shape[1])
            elif(i == len(bottom_1xn) - 1):
                top = top.reshape(shape[0], 1, shape[1])
            
            false_contracted = np.einsum("ler, LRe -> lLrR", bottom, top)

            contracted_left = false_contracted.shape[0] * false_contracted.shape[1]
            contracted_right = false_contracted.shape[2] * false_contracted.shape[3]

            contracted = false_contracted.reshape(contracted_left,
                                                  contracted_right)
            contracted_1xn.append(contracted)
        
        current = np.matmul(contracted_1xn[0], contracted_1xn[1])
        for i in range(2, len(contracted_1xn)):
            current = np.matmul(current, contracted_1xn[i])

        return current

    log_scale_accumulator = 0.0
    total_error_sq = 0.0

    bottom_1xn = tensor_list[0:n]

    for i in range(1, n - 1):
        top_1xn = tensor_list[n * i:n * (i + 1)]
        bottom_1xn, step_error = modular_2x_1xn_contraction(bottom_1xn, top_1xn)

        total_error_sq += step_error**2

        generate_left_canonical_form(bottom_1xn)

        last_tensor = bottom_1xn[-1]
        norm = np.linalg.norm(last_tensor)

        if norm > 0:
            bottom_1xn[-1] /= norm
            log_scale_accumulator += np.log(norm)

    top_1xn = tensor_list[-n:]
    final_small = float(np.squeeze(final_contraction(bottom_1xn, top_1xn)))

    logZ = np.log(final_small) + log_scale_accumulator
    exponent = int(np.floor(logZ / np.log(10)))
    mantissa = np.exp(logZ - exponent * np.log(10))

    total_error = np.sqrt(total_error_sq)

    if(n < 23):
        relative_error = total_error / (mantissa * 10**exponent) # Wrong
    else:
        relative_error = total_error # Impossible case?

    return mantissa, exponent, total_error, relative_error

if __name__ == "__main__":
    n: int = int(input("Number of Rows of Tensor List "))
    tensor_list = make_n_by_n_network(n)
    for tensor in tensor_list:
        print(tensor)
    if(n <= 10):
        result_uncompressed = contract_n_by_n_network(tensor_list, n)
        print("Result of uncompressed contraction: \n", result_uncompressed)
    result_compressed, exp, terr, rerr = contract_n_by_n_network(tensor_list, n, compress=True, dim=2)
    result_compressed_2, exp2, terr2, rerr2 = contract_n_by_n_network(tensor_list, n, compress=True, lose_dim=6)
    result_compressed_3, exp3, terr3, rerr3 = contract_n_by_n_network(tensor_list, n, compress=True, comp_percent=.34)
    print("Result of compressed contraction: \n", result_compressed)
    print("Result of second compressed contraction: \n", result_compressed_2)
    print("Result of third compressed contraction: \n", result_compressed_3)
    print("Exponent of compressed contraction: \n", exp)
    print("Exponent of second compressed contraction: \n", exp2)
    print("Exponent of third compressed contraction: \n", exp3)
    print("Total Error of first compressed contraction: \n", terr)
    print("Total Error of second compressed contraction: \n", terr2)
    print("Total Error of third compressed contraction: \n", terr3)
    print("Relative Error of first compressed contraction: \n", rerr)
    print("Relative Error of second compressed contraction: \n", rerr2)
    print("Relative Error of third compressed contraction: \n", rerr3)
