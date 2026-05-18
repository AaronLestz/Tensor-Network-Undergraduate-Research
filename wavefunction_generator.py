import numpy as np

def generate_all_spin_up_network(bond_dim=2, phys_dim=2):
    """
    Generates an all spin-up wave function tensor network.
    The spin-up state |↑> is represented by the vector [1.0, 0.0].
    """
    psi_net, psi_conj_net = [], []
    
    for _ in range(9):
        # Initialize tensor shape: (Left, Up, Right, Down, Physical)
        psi = np.zeros((bond_dim, bond_dim, bond_dim, bond_dim, phys_dim), dtype=complex)
        psi_conj = np.zeros((bond_dim, bond_dim, bond_dim, bond_dim, phys_dim), dtype=complex)
        
        psi[0, 0, 0, 0, 0] = 1.0       # Spin up amplitude
        psi_conj[0, 0, 0, 0, 0] = 1.0  # Complex conjugate is same b/c just reals
        
        psi_net.append(psi)
        psi_conj_net.append(psi_conj)
        
    return psi_net, psi_conj_net