import torch
from torch import nn

def transpose_qkv(X, num_heads):
    '''将单个注意力拆分为h个头'''
    # 输入X形状(batch, seq_length, attn_size)
    batch_size, seq_length, attn_size = X.shape
    head_dim = attn_size // num_heads
    # 将attn_size拆分为num_heads和head_dim
    X = X.reshape(batch_size, seq_length, num_heads, head_dim)
    # 形状变为(batch * num_heads, seq_length, head_dim) 
    X = X.transpose(1, 2).reshape(batch_size * num_heads, seq_length, head_dim)  
    return X