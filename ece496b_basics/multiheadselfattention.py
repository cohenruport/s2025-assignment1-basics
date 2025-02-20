import torch
from torch import nn

from ece496b_basics.softmax import softmax

class multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop=0.0):
        super(multihead_self_attention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.dv = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(attn_pdrop)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.dv).transpose(1, 2)
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float32, device=x.device))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attention_weights = softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attention_output)
        
        return output