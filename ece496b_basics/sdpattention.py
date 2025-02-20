import numpy as np

from ece496b_basics.softmax import softmax

def scaled_dot_product_attention(Q, K, V, mask=None, dropout_rate=0.0):
    dk = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(-1, -2)) / np.sqrt(dk)
    
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)
    
    attention_weights = softmax(scores, dim=-1)
    
    if dropout_rate > 0.0:
        dropout_mask = np.random.rand(*attention_weights.shape) > dropout_rate
        attention_weights *= dropout_mask
        attention_weights /= (1.0 - dropout_rate)
    
    output = np.matmul(attention_weights, V)
    
    return output