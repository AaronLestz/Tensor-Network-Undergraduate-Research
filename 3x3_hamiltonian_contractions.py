import copy
import numpy as np
from wavefunction_generator import generate_all_spin_up_network
from model_adapter import adapt_gmpos_to_hamiltonian_net
from gmpos.models.heis_zz.ham import get_gmpos_zz

def generate_tensor_networks(bond_dim=2, phys_dim=2):
    """
    Generates the psi, H, and psi* tensors for a 3x3 grid.
    """

    psi_net, psi_conj_net, H_net = [], [], []

    for _ in range(9):
        # (left, up, right, down, out)
        psi = np.random.rand(bond_dim, bond_dim, bond_dim, bond_dim, phys_dim)
        psi_conj = np.random.rand(bond_dim, bond_dim, bond_dim, bond_dim, phys_dim)

        # (left, up, right, down, out_ket, out_bra)
        H = np.random.rand(bond_dim, bond_dim, bond_dim, bond_dim, phys_dim, phys_dim)
        for p in range(phys_dim):
            H[0, 0, 0, 0, p, p] = 1.0 # Identity path

        psi_net.append(psi)
        psi_conj_net.append(psi_conj)
        H_net.append(H)

    return psi_net, H_net, psi_conj_net

def contract_triple_layer(t_psi, t_H, t_psi_star):
    res = np.einsum('abcdn, efghmn, ABCDm -> aAebBfcCgdDh', 
                    t_psi, t_H, t_psi_star, optimize=True)
    s = res.shape
    return res.reshape(s[0]*s[1]*s[2], s[3]*s[4]*s[5], s[6]*s[7]*s[8], s[9]*s[10]*s[11])

def get_identity_H(phys_dim):
    I = np.eye(phys_dim)
    return I.reshape(1, 1, 1, 1, phys_dim, phys_dim)

def reshape_edge_tensors(H_tensor, L=False, U=False, R=False, D=False):
    if(not L):
        H_tensor = H_tensor[:1, :, :, :, :, :]
    if(not U):
        H_tensor = H_tensor[:, :1, :, :, :, :]
    if(not R):
        H_tensor = H_tensor[:, :, :1, :, :, :]
    if(not D):
        H_tensor = H_tensor[:, :, :, :1, :, :]
    return H_tensor

def contract_1x3_hamiltonian(psi_net, H_net, psi_star_net):
    fused, p_dim = [], psi_net[0].shape[-1]
    for i in range(9):
        if(i in [0, 3, 6]):
            H = reshape_edge_tensors(H_net[i], U=(i in [0, 3]), D=(i in [3, 6]))
        else:
            H = get_identity_H(p_dim)
        fused.append(contract_triple_layer(psi_net[i], H, psi_star_net[i]))
    return fused

def contract_2x3_hamiltonian(psi_net, H_net, psi_star_net):
    fused, p_dim = [], psi_net[0].shape[-1]
    for i in range(9):
        if(i in [0, 3, 6]): 
            H = reshape_edge_tensors(H_net[i], R=True)
        elif(i in [1, 4, 7]): 
            H = reshape_edge_tensors(H_net[i], L=True, U=(i in [1, 4]), D=(i in [4, 7]))
        else:
            H = get_identity_H(p_dim)
        fused.append(contract_triple_layer(psi_net[i], H, psi_star_net[i]))
    return fused

def contract_3x3_hamiltonian(psi_net, H_net, psi_star_net):
    fused, p_dim = [], psi_net[0].shape[-1]
    for i in range(9):
        H = reshape_edge_tensors(H_net[i], L=(i in [1, 2, 4, 5, 7, 8]), 
                                 R=(i in [0, 1, 3, 4, 6, 7]), U=(i in [2, 5]), D=(i in [5, 8]))
        fused.append(contract_triple_layer(psi_net[i], H, psi_star_net[i]))
    return fused

def adapt_to_2d_contraction(fused_net):
    """
    Takes off the extra dimensions (of L, U, R, or D) in order to match
    the expected input for the (previously made) 2d contraction algorithm
    """
    t = fused_net
    adapted = []
    # Only 8 b/c middle needed to be 4D regardless
    adapted.append(t[0][0, :, :, 0])    # 0: (U, R)
    adapted.append(t[1][:, :, :, 0])    # 1: (L, U, R)
    adapted.append(t[2][:, :, 0, 0])    # 2: (L, U)
    adapted.append(t[3][0, :, :, :])    # 3: (U, R, D)
    adapted.append(t[4][:, :, :, :])    # 4: (L, U, R, D)
    adapted.append(t[5][:, :, 0, :])    # 5: (L, U, D)
    adapted.append(t[6][0, 0, :, :])    # 6: (R, D)
    adapted.append(t[7][:, 0, :, :])    # 7: (L, R, D)
    adapted.append(t[8][:, 0, 0, :])    # 8: (L, D)

    return adapted


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
    """Calculates L^2 Error between exact & compressed mps"""
    def mps_inner_product(mps1, mps2):
        E = np.einsum("lpr, lpR -> rR", mps1[0].conj(), mps2[0])
        for i in range(1, len(mps1)):
            E = np.einsum("lL, lpr, LpR -> rR", E, mps1[i].conj(), mps2[i])
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
                else: return tensor
            else:
                if(order == 2):
                    if(index == 0):
                        return tensor.reshape(1, shape[0], shape[1])
                    else: 
                        return tensor.reshape(shape[0], shape[1], 1)
                else: return tensor
                
        contracted_1xn = []
        for i, bottom_tensor in enumerate(bottom_1xn):
            bottom = set_proper_tensor_orders(bottom_tensor, i, False)
            top = set_proper_tensor_orders(top_1xn[i], i, True)
            false_contracted: np.ndarray = np.einsum("ler, LURe -> lLUrR", bottom, top)
            contracted_left = false_contracted.shape[0] * false_contracted.shape[1]
            contracted_right = false_contracted.shape[3] * false_contracted.shape[4]
            contracted = false_contracted.reshape(contracted_left, false_contracted.shape[2], contracted_right)
            contracted_1xn.append(contracted)

        if(compress):
            exact_copy = copy.deepcopy(contracted_1xn)
            mps_svd_contraction(contracted_1xn, dim, lose_dim, comp_percent)
            step_error = compute_mps_error(exact_copy, contracted_1xn)
            return contracted_1xn, step_error
        return contracted_1xn, 0
    
    def final_contraction(bottom_1xn: list, top_1xn: list):
        contracted_1xn = []
        for i, _ in enumerate(bottom_1xn):
            bottom = bottom_1xn[i]
            top = top_1xn[i]
            shape = top.shape
            if(i == 0):
                top = top.reshape(1, shape[0], shape[1])
            elif(i == len(bottom_1xn) - 1):
                top = top.reshape(shape[0], 1, shape[1])

            false_contracted = np.einsum("ler, LRe -> lLrR", bottom, top)
            contracted_left = false_contracted.shape[0] * false_contracted.shape[1]
            contracted_right = false_contracted.shape[2] * false_contracted.shape[3]
            contracted = false_contracted.reshape(contracted_left, contracted_right)
            contracted_1xn.append(contracted)

        current_contracted_state = np.matmul(contracted_1xn[0], contracted_1xn[1])
        for i in range(2, len(contracted_1xn)):
            current_contracted_state = np.matmul(current_contracted_state, contracted_1xn[i])
        return current_contracted_state

    total_error_sq = 0.0
    bottom_1xn = tensor_list[0:n]
    for i in range(1, n - 1):
        top_1xn = tensor_list[n * i:n * (i + 1)]
        bottom_1xn, step_error = modular_2x_1xn_contraction(bottom_1xn, top_1xn)
        total_error_sq += step_error**2
    top_1xn = tensor_list[-n:]
    total_error = np.sqrt(total_error_sq)
    final_contracted_value = np.squeeze(final_contraction(bottom_1xn, top_1xn))
    relative_error = total_error / final_contracted_value if final_contracted_value != 0 else 0
    return final_contracted_value, total_error, relative_error

def create_and_contract():
    # psi, H, psi_star = generate_tensor_networks()
    psi, psi_star = generate_all_spin_up_network()

    bot_dict, vert_dict, gmpo_dict = get_gmpos_zz(3, 3)
    print(bot_dict['zz'][0].shape)
    H = adapt_gmpos_to_hamiltonian_net(bot_dict, vert_dict, gmpo_dict, 3, 3)

    raw_H1 = contract_1x3_hamiltonian(psi, H, psi_star)
    raw_H2 = contract_2x3_hamiltonian(psi, H, psi_star)
    raw_H3 = contract_3x3_hamiltonian(psi, H, psi_star)

    tn_H1 = adapt_to_2d_contraction(raw_H1)
    tn_H2 = adapt_to_2d_contraction(raw_H2)
    tn_H3 = adapt_to_2d_contraction(raw_H3)

    res1, t_err1, r_err1 = contract_n_by_n_network(tn_H1, n=3, compress=True, dim=4)
    res2, t_err2, r_err2 = contract_n_by_n_network(tn_H2, n=3, compress=True, dim=4)
    res3, t_err3, r_err3 = contract_n_by_n_network(tn_H3, n=3, compress=True, dim=4)

    print(f"Hamiltonian 1 has Value: {res1:e}, Error: {t_err1:e}, and Relative Error: {r_err1:e}")
    print(f"Hamiltonian 2 has Value: {res2:e}, Error: {t_err2:e}, and Relative Error: {r_err2:e}")
    print(f"Hamiltonian 3 has Value: {res3:e}, Error: {t_err3:e}, and Relative Error: {r_err3:e}")
    print(f"Total Value = {res1 + res2 + res3}")

if __name__ == "__main__":
    create_and_contract()
