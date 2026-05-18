import numpy as np

def adapt_gmpos_to_hamiltonian_net(bot_dict, vert_dict, gmpo_dict, Lx, Ly):
    """
    Converts dictionaries from GMPOS to my contraction convention
    within the 3x3_hamiltonian_contraction file
    """

    H_net = [None] * (Lx * Ly)
    
    for gen_y in range(Ly):
        for gen_x in range(Lx):
            
            cont_x = (Ly - 1) - gen_y
            cont_y = gen_x
            
            # List index
            idx = cont_y * Lx + cont_x
            
            if gen_y == 0:
                H_h = bot_dict['zz'][gen_x].copy()
            else:
                H_h = gmpo_dict[(gen_y, 'zz')][gen_x].copy()

            if gen_y < Ly - 1:
                H_v = vert_dict[(gen_x, 'zz')][gen_y].copy()
                
                H_v_sliced = H_v[:, 0, :, :]
                
                T = np.einsum('ldrop, upi -> urdloi', H_h, H_v_sliced)
                
            else:
                T = H_h.transpose(2, 1, 0, 3, 4)[None, :, :, :, :, :]
                
            H_net[idx] = T

    return H_net