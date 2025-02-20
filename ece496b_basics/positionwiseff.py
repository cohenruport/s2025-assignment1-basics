import numpy as np

from ece496b_basics.gelu import gelu_approx


def positionwise_feedforward(x, W1, W2):
    x_transformed = np.dot(x, W1)
    x_activated = gelu_approx(x_transformed)
    
    output = np.dot(x_activated, W2)
    
    return output