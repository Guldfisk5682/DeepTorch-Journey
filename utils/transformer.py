import torch
from torch import nn
import math

def transpose_qkv(X, num_heads):
    '''将单个注意力拆分为h个头, Input shape:(batch,seq_len,d_model)'''
    assert X.shape[2] % num_heads == 0, "d_model 必须能被 num_heads 整除"
    X=X.reshape(X.shape[0],X.shape[1],num_heads,-1) # 变为(batch,seq_len,num_heads,head_dim)
    X=X.transpose(1,2) # 变为(batch,num_heads,seq_len,head_dim)
    return X.reshape(-1,X.shape[2],X.shape[3]) # 变为(batch*num_heads,seq_len,head_dim)

def transpose_output(X,num_heads):
    '''合并多头输出,输入形状为(batch*seq_len,num_heads,head_dim)'''
    X=X.reshape(-1,num_heads,X.shape[1],X.shape[2]) # 拆分回四维
    X=X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1) # 变为(batch,seq_len,d_model)

def masked_softmax(X,valid_len):
    '''
    变长序列遮蔽处理,valid_len为(bacth,seq_len)或(batch,)
    输入X形状为(batch,q_seq_len,k_seq_len)
    '''
    if valid_len is None:
        return torch.softmax(X,dim=-1)
    mask=torch.ones_like(X,dtype=torch.bool)
    if valid_len.dim()==1:
        valid_len=valid_len[:,None]
    # 在最后一个维度实施mask,因为最后一个维度是key_seql_len(长度为seq_len)
    # 遮蔽的目的是掩盖掉无效的key_seql_len
    mask[:,:,:valid_len.max()]=False
    X_masked=X.masked_fill(mask,-1e6)
    return torch.softmax(X_masked,dim=-1)

class DotProductAttention(nn.Module):
    '''缩放点积注意力'''
    def __init__(self,dropout) -> None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,queries,keys,values,valid_len=None):
        # 输入数据形状为(batch,seq_len,d_model)
        d=queries.shape[-1]
        # scores形状为(batch,q_seq_len,k_seq_len) 两个seq_len形状相同意义不同
        scores=torch.bmm(queries,keys.transpose(1,2))/math.sqrt(d)
        self.attention_weights=masked_softmax(scores,valid_len)
        return torch.bmm(self.dropout(self.attention_weights),values)

class MultiHeadAttention(nn.Module):
    '''多头注意力'''
    def __init__(self,d_model,
                 num_heads,dropout):
        super().__init__()
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.attention=DotProductAttention(dropout)
        self.W_q=nn.Linear(d_model,d_model,bias=False)
        self.W_k=nn.Linear(d_model,d_model,bias=False)
        self.W_v=nn.Linear(d_model,d_model,bias=False)
        self.W_o=nn.Linear(d_model,d_model,bias=False)
    
    def forward(self,queries,keys,values,valid_len=None):
        # 处理后数据形状变为(batch * num_heads, seq_length, head_dim)
        q_trans = transpose_qkv(self.W_q(queries), self.num_heads)
        k_trans = transpose_qkv(self.W_k(keys), self.num_heads)
        v_trans = transpose_qkv(self.W_v(values), self.num_heads)
        
        # 这里使用repeat_interleave而不是repeat是因为
        # 每个batch有num_heads个头,我们的目的是在batch上应用掩码
        # 若使用repeat会将掩码交错应用
        if valid_len is not None:
            valid_len=torch.repeat_interleave(valid_len,self.num_heads,dim=0)

        attention_output = self.attention(q_trans, k_trans, v_trans, valid_len)
        output=transpose_output(attention_output,self.num_heads)
        return self.W_o(output)

class PositionEncoding(nn.Module):
    '''位置编码'''
    def __init__(self,d_model,dropout,max_len=1000) -> None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        # 多一个1维度便于批处理
        self.PN=torch.zeros(1,max_len,d_model)
        # 形状变为(max_len,1)
        pos=torch.arange(max_len,dtype=torch.float32).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        argument = pos / div_term
        self.PN[:,:,0::2]=torch.sin(argument)
        self.PN[:,:,1::2]=torch.cos(argument)
        
    def forward(self,X):
        X=X+self.PN[:,:X.shape[1],:].to(X.device)
        return self.dropout(X)
    
class AddNorm(nn.Module):
    '''残差连接和层规范化'''
    def __init__(self,d_model,dropout) -> None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.LN=nn.LayerNorm(d_model)
    
    def forward(self,Y,X):
        return self.dropout(self.LN(X+Y))

class PositionWiseFFN(nn.Module):
    '''基于位置的前馈神经网络'''
    def __init__(self,d_model,ffn_hidden) -> None:
        # ffn_hidden一般为d_model的4倍
        super().__init__()
        self.dense1=nn.Linear(d_model,ffn_hidden)
        self.relu=nn.ReLU()
        self.dense2=nn.Linear(ffn_hidden,d_model)
        
    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))
    
class EncoderBlock(nn.Module):
    '''Transformer编码器块'''
    def __init__(self,d_model,ffn_hidden,num_heads,dropout) -> None:
        super().__init__()
        self.attention=MultiHeadAttention(d_model,num_heads,dropout)
        self.ffn=PositionWiseFFN(d_model,ffn_hidden)
        self.addnorm1=AddNorm(d_model,dropout)
        self.addnorm2=AddNorm(d_model,dropout)

    def forward(self,X,valid_len):
        Y=self.addnorm1(self.attention(X,X,X,valid_len),X)
        return self.addnorm2(self.ffn(Y),Y)
    
class DecoderBlock(nn.Module):
    '''Transformer解码器块'''
    def __init__(self,d_model,ffn_hidden,num_heads,dropout) -> None:
        super().__init__()
        self.masked_attention=MultiHeadAttention(d_model,num_heads,dropout)
        self.attention=MultiHeadAttention(d_model,num_heads,dropout)
        self.ffn=PositionWiseFFN(d_model,ffn_hidden)
        self.addnorm1=AddNorm(d_model,dropout)
        self.addnorm2=AddNorm(d_model,dropout)
        self.addnorm3=AddNorm(d_model,dropout)
    
    def forward(self,X,enc_output,valid_len=None,causal_mask=None):
        # 遮蔽未来信息
        X=self.addnorm1(self.masked_attention(X,X,X,causal_mask),X)
        X=self.addnorm2(self.attention(X,enc_output,enc_output,valid_len),X)
        return self.addnorm3(self.ffn(X),X)

class Transformer(nn.Module):
    def __init__(self,vocab_size,num_layers=6, d_model=512, ffn_hidden=2048, num_heads=8, dropout=0.1):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.pos_encoding = PositionEncoding(d_model, dropout)
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model,ffn_hidden,num_heads,dropout)
            for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model,ffn_hidden,num_heads,dropout)
            for _ in range(num_layers)
        ])
        self.linear=nn.Linear(d_model,vocab_size)
    
    def forward(self, enc_input, dec_input, valid_len=None, causal_mask=None):
        # 应用嵌入和位置编码
        enc_input = self.pos_encoding(self.embedding(enc_input))
        dec_input = self.pos_encoding(self.embedding(dec_input))
        
        # 编码器前向传播
        enc_output = enc_input
        for encoder_block in self.encoder:
            enc_output = encoder_block(enc_output, valid_len)
            
        # 解码器前向传播
        dec_output = dec_input
        for decoder_block in self.decoder:
            dec_output = decoder_block(dec_output, enc_output, valid_len, causal_mask)
            
        return self.linear(dec_output)