import torch
import torch.nn as nn

from ece496b_basics.RMSnorm import RMSNorm
from ece496b_basics.multiheadselfattention import multihead_self_attention
from ece496b_basics.positionwiseff import positionwise_feedforward

class transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop):
        super(transformer_block, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop

        # Define the layers
        self.rmsnorm1 = RMSNorm(d_model)
        self.multihead_self_attention = multihead_self_attention(d_model, num_heads, attn_pdrop)
        self.dropout1 = nn.Dropout(residual_pdrop)

        self.rmsnorm2 = RMSNorm(d_model)
        self.positionwise_feedforward = positionwise_feedforward(d_model, d_ff)
        self.dropout2 = nn.Dropout(residual_pdrop)

    def forward(self, x):
        # First sublayer: Multi-head self-attention
        norm_x1 = self.rmsnorm1(x)
        attn_output = self.multihead_self_attention(norm_x1)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output  # Residual connection

        # Second sublayer: Position-wise feed-forward network
        norm_x2 = self.rmsnorm2(x)
        ff_output = self.positionwise_feedforward(norm_x2)
        ff_output = self.dropout2(ff_output)
        x = x + ff_output  # Residual connection

        return x