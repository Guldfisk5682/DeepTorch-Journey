import torch
from torch import nn
import math

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

def sequence_mask(X,valid_len,value=0):
    maxlen=X.size(1)
    mask=torch.arange(maxlen,device=X.device)[None,:]<valid_len[:,None]
    # 将False位置的值设置为value
    X[~mask]=value
    return X

def masked_softmax(X,valid_len):
    if valid_len is None:
        return torch.softmax(X)
    else:
        X=sequence_mask(X,valid_len,value=-1e5)
        return torch.softmax(X,dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self,dropout) -> None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,queries,keys,values,valid_len=None):
        # 缩放点积注意力的数据形状相同
        d=queries.shape[-1]
        scores=torch.bmm(queries,keys.transpose(1,2))/math.sqrt(d)
        self.attention_weights=masked_softmax(scores,valid_len)
        return torch.bmm(self.dropout(self.attention_weights),values)

class MultiHeadAttention(nn.Module):
    def __init__(self,query_size,key_size,value_size,attn_size,
                 num_heads,dropout):
        super().__init__()
        self.num_heads=num_heads
        self.attention=DotProductAttention(dropout)
        self.W_q=nn.Linear(query_size,attn_size,bias=False)
        self.W_k=nn.Linear(key_size,attn_size,bias=False)
        self.W_v=nn.Linear(value_size,attn_size,bias=False)
        self.W_o=nn.Linear(attn_size,attn_size,bias=False)
    
    def forward(self,queries,keys,values,valid_len=None):
        # 处理后数据形状变为(batch, num_heads, seq_length, head_dim)
        q_trans=transpose_qkv(self.W_q(queries),self.num_heads)
        k_trans = transpose_qkv(self.W_k(keys), self.num_heads)
        v_trans = transpose_qkv(self.W_v(values), self.num_heads)
        
        if valid_len is not None:
            valid_len=torch.repeat_interleave(valid_len,self.num_heads)
           
        attention_output = self.attention(q_trans, k_trans, v_trans, valid_len)
        batch_size = queries.shape[0]
        seq_length_q = q_trans.shape[1] 
        head_dim = attention_output.shape[2] 
        # 变回四维
        output = attention_output.reshape(batch_size, self.num_heads, seq_length_q, head_dim)

        output = output.permute(0, 2, 1, 3) 
        # 合并头
        attn_size = self.num_heads * head_dim 
        output = output.reshape(batch_size, seq_length_q, attn_size)
        return self.W_o(output)

class PositionEncoding(nn.Module):
    '''位置编码'''
    def __init__(self,d,dropout,max_len=1000) -> None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        # 多一个1维度便于批处理
        self.PN=torch.zeros(1,max_len,d)
        pos=torch.arange(max_len,dtype=torch.float32)
        pos=pos/torch.pow(10000,torch.arange(0,d,2,dtype=torch.float32)/d)
        self.PN[:,:,0::2]=torch.sin(pos)
        self.PN[:,:,1::2]=torch.cos(pos)
        
    def forward(self,X):
        X=X+self.PN[:,:X.shape[1],:]
        return self.dropout(X)
    
class AddNorm(nn.Module):
    '''残差连接和层规范化'''
    def __init__(self,norm_size,dropout) -> None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.LN=nn.LayerNorm(norm_size)
    
    def forward(self,Y,X):
        return self.dropout(self.LN(X+Y))

class PositionWiseFFN(nn.Module):
    '''基于位置的前馈神经网络'''
    def __init__(self,ffn_input,ffn_hidden,ffn_output) -> None:
        super().__init__()
        self.dense1=nn.Linear(ffn_input,ffn_hidden)
        self.relu=nn.ReLU()
        self.dense2=nn.Linear(ffn_hidden,ffn_output)
        
    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))
    
class EncoderBlock(nn.Module):
    '''Transformer编码器块'''
    def __init__(self,query_size,key_size,value_size,attn_size,norm_size,
                 ffn_input,ffn_hidden,num_heads,dropout) -> None:
        super().__init__()
        self.attention=MultiHeadAttention(query_size,key_size,value_size,
                                          attn_size,num_heads,dropout)
        self.ffn=PositionWiseFFN(ffn_input,ffn_hidden,attn_size)
        self.addnorm=AddNorm(norm_size,dropout)

    def forward(self,X,valid_len):
        Y=self.addnorm(self.attention(X,X,X,valid_len),X)
        return self.addnorm(self.ffn(Y),Y)