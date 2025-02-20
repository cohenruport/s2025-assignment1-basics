import numpy as np

def gelu_approx(x):
    # Constants for the approximation
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    c = 0.044715
    
    # Compute the approximation
    x_cubed = x ** 3
    tanh_arg = sqrt_2_over_pi * (x + c * x_cubed)
    return 0.5 * x * (1.0 + np.tanh(tanh_arg))