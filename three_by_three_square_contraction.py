import numpy as np

# Defining proper direction order as (left, up right, down)

def make_three_by_three_network():
    # 0 - 4 - 1
    # 5 - 8 - 6
    # 3 - 7 - 2
    tensor_list = []
    for i in range(9):
        if(i < 4):
            mat = np.random.rand(2,2)
        elif(i < 8):
            mat = np.random.rand(2,2,2)
        else:
            mat = np.random.rand(2,2,2,2)
        tensor_list.append(mat)
    return tensor_list

def contract_three_by_three_network(tensor_list: list):
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
            false_contracted = np.einsum("ler, LURe -> lLUrR", bottom, top)
            contracted_left = false_contracted.shape[0] * false_contracted.shape[1]
            contracted_right = false_contracted.shape[3] * false_contracted.shape[4]
            contracted = false_contracted.reshape(contracted_left, false_contracted.shape[2], contracted_right)
            contracted_1x3.append(contracted)
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


    bottom_1x3 = [tensor_list[3], tensor_list[7], tensor_list[2]]
    top_1x3 = [tensor_list[5], tensor_list[8], tensor_list[6]]
    bottom_1x3 = modular_2x_1x3_contraction(bottom_1x3, top_1x3)
    top_1x3 = [tensor_list[0], tensor_list[4], tensor_list[1]]
    return np.squeeze(final_contraction(bottom_1x3, top_1x3))


if __name__ == "__main__":
    tensor_list = make_three_by_three_network()
    for tensor in tensor_list:
        print(tensor)
    result = contract_three_by_three_network(tensor_list)
    print("Result of contraction: \n", result)