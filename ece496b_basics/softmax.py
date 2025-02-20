import numpy as np

def softmax(tensor, dim):
    max_vals = np.max(tensor, axis=dim, keepdims=True)
    stable_tensor = tensor - max_vals
    
    exp_tensor = np.exp(stable_tensor)
    
    sum_exp = np.sum(exp_tensor, axis=dim, keepdims=True)
    
    softmax_tensor = exp_tensor / sum_exp
    
    return softmax_tensor