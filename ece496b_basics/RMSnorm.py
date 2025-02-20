import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, weights=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        # Initialize the gain parameter (gi) using the weights dictionary
        if weights and 'weight' in weights:
            self.gain = nn.Parameter(weights['weight'])
        else:
            self.gain = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Compute the RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize the input
        x_normalized = x / rms
        # Apply the gain parameter
        return self.gain * x_normalized