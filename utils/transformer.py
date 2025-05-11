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
    '''合并多头输出,输入形状为(batch*num_heads,seq_len,head_dim)'''
    # X.shape: batch*num_heads,seq_len,head_dim
    original_batch_size = X.shape[0] // num_heads
    seq_len = X.shape[1]
    head_dim = X.shape[2]
    
    X=X.reshape(original_batch_size, num_heads, seq_len, head_dim) # (batch, num_heads, seq_len, head_dim)
    X=X.permute(0,2,1,3) # (batch, seq_len, num_heads, head_dim)
    return X.reshape(X.shape[0],X.shape[1],-1) # 变为(batch,seq_len,d_model)

def masked_softmax(X, valid_len_or_mask_tensor, is_causal_scenario=False):
    """
    在X的最后一个维度上执行 softmax 操作，支持基于有效长度的填充掩码或直接的布尔因果掩码。
    X shape: (batch_eff, q_seq_len, k_seq_len)   * batch_eff==batch*num_heads
    valid_len_or_mask_tensor:
        - if is_causal_scenario=True:  它是布尔因果掩码张量 (batch_eff, q_seq_len, k_seq_len)，True表示掩码。
        - if is_causal_scenario=False: 它是表示key序列有效长度的张量 (batch_eff,) 或 (batch_eff, 1)。
                                     如果为None，则不应用掩码。
    is_causal_scenario: 布尔标志，指示是否应用因果掩码逻辑。
    
    对于valid_len的形状说明:
        -  (batch_eff,):其中每个元素代表对应批次项（在这个有效批次中）的序列的有效长度。
            如对于valid_len = torch.tensor([5, 3])这意味着第0/1个batch中k_len的有效长度是 5/3。
        - (batch_eff, 1):其中每行代表对应批次项，该行只有一个元素，这个元素就是该序列的有效长度。
            如valid_len = torch.tensor([[5], [3]]) 第 0 个batch（即 valid_len[0]）的有效长度是 valid_len[0, 0] = 5
    """
    if is_causal_scenario:
        # valid_len_or_mask_tensor 是预先计算好的因果掩码张量
        # True 值表示需要被掩码的位置
        causal_mask_tensor = valid_len_or_mask_tensor.to(device=X.device, dtype=torch.bool)
        X_masked = X.masked_fill(causal_mask_tensor, -1e6)
    elif valid_len_or_mask_tensor is not None:
        # 处理基于 valid_len (有效长度) 的填充掩码
        valid_lengths = valid_len_or_mask_tensor # (batch_eff,) or (batch_eff, 1)
        k_seq_len = X.shape[-1] # key 序列的长度

        if valid_lengths.dim() == 1:
            valid_lengths = valid_lengths.unsqueeze(1) # -> (batch_eff, 1)
            
        # 代表了 Key 序列中所有可能的位置索引
        indices = torch.arange(k_seq_len, device=X.device).unsqueeze(0).unsqueeze(0) # -> (1, 1, k_seq_len)
        padding_mask = indices >= valid_lengths.unsqueeze(-1) # -> (batch_eff, 1, k_seq_len)
        
        final_padding_mask = padding_mask.expand_as(X) # -> (batch_eff, q_seq_len, k_seq_len)
        X_masked = X.masked_fill(final_padding_mask, -1e6)
    else:
        # 没有掩码 (valid_len_or_mask_tensor is None and not is_causal_scenario)
        return torch.softmax(X, dim=-1)
        
    return torch.softmax(X_masked, dim=-1)

def create_causal_mask(seq_len,batch_size,num_heads, device=torch.device('cpu')):
    '''创建因果掩码'''
    causal_mask=torch.triu(torch.ones(seq_len,seq_len, device=device),diagonal=1).bool()
    causal_mask=causal_mask.unsqueeze(0).expand(batch_size*num_heads,-1,-1)
    return causal_mask

class DotProductAttention(nn.Module):
    '''缩放点积注意力'''
    def __init__(self,dropout) -> None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,queries,keys,values,valid_len_or_mask_tensor=None,is_causal_scenario=False):
        d=queries.shape[-1]
        scores=torch.bmm(queries,keys.transpose(1,2))/math.sqrt(d)
        self.attention_weights=masked_softmax(scores,valid_len_or_mask_tensor,is_causal_scenario)
        return torch.bmm(self.dropout(self.attention_weights),values)

class MultiHeadAttention(nn.Module):
    '''多头注意力'''
    def __init__(self,d_model,
                 num_heads,dropout):
        super().__init__()
        self.num_heads=num_heads
        self.attention=DotProductAttention(dropout)
        self.W_q=nn.Linear(d_model,d_model,bias=False)
        self.W_k=nn.Linear(d_model,d_model,bias=False)
        self.W_v=nn.Linear(d_model,d_model,bias=False)
        self.W_o=nn.Linear(d_model,d_model,bias=False)
    
    def forward(self,queries,keys,values,valid_len=None,is_causal_scenario=False):
        q_trans = transpose_qkv(self.W_q(queries), self.num_heads)
        k_trans = transpose_qkv(self.W_k(keys), self.num_heads)
        v_trans = transpose_qkv(self.W_v(values), self.num_heads)
        
        # 根据是因果场景还是填充场景来处理 valid_len 或 mask_tensor
        # is_causal_scenario=True: valid_len 就是预计算的因果掩码张量, 无需repeat
        # is_causal_scenario=False: valid_len 是长度张量, 需要为多头repeat
        if not is_causal_scenario and valid_len is not None:
            # 这是处理填充掩码的情况，valid_len 是长度张量
            processed_mask_input = torch.repeat_interleave(valid_len, self.num_heads, dim=0)
        else:
            # 这是因果掩码情况 (valid_len 是 causal_mask_tensor) 或 valid_len is None
            processed_mask_input = valid_len

        attention_output = self.attention(q_trans, k_trans, v_trans, processed_mask_input, is_causal_scenario)
        output=transpose_output(attention_output,self.num_heads)
        return self.W_o(output)

class PositionEncoding(nn.Module):
    '''位置编码'''
    def __init__(self,d_model,dropout,max_len=1000) -> None:
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        pe = torch.zeros(1, max_len, d_model)
        pos=torch.arange(max_len,dtype=torch.float32).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        argument = pos / div_term
        pe[:, :, 0::2] = torch.sin(argument)
        pe[:, :, 1::2] = torch.cos(argument)
        self.register_buffer('PN', pe)
        
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
        Y=self.addnorm1(self.attention(X,X,X,valid_len,is_causal_scenario=False),X)
        return self.addnorm2(self.ffn(Y),Y)
    
class DecoderBlock(nn.Module):
    '''Transformer解码器块'''
    def __init__(self,d_model,ffn_hidden,num_heads,dropout) -> None:
        super().__init__()
        self.masked_attention=MultiHeadAttention(d_model,num_heads,dropout) # Self-attention
        self.attention=MultiHeadAttention(d_model,num_heads,dropout)      # Cross-attention
        self.ffn=PositionWiseFFN(d_model,ffn_hidden)
        self.addnorm1=AddNorm(d_model,dropout)
        self.addnorm2=AddNorm(d_model,dropout)
        self.addnorm3=AddNorm(d_model,dropout)
    
    def forward(self,X,enc_output,cross_attention_valid_len=None,self_attention_causal_mask=None):
        Y = self.masked_attention(X, X, X, valid_len=self_attention_causal_mask, is_causal_scenario=True)
        X_after_self_attn = self.addnorm1(Y, X)
        
        Y = self.attention(X_after_self_attn, enc_output, enc_output, valid_len=cross_attention_valid_len, is_causal_scenario=False)
        X_after_cross_attn = self.addnorm2(Y, X_after_self_attn)
        
        return self.addnorm3(self.ffn(X_after_cross_attn), X_after_cross_attn)

class Transformer(nn.Module):
    '''Transformer'''
    def __init__(self,vocab_size,num_layers=6, d_model=512, ffn_hidden=2048, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads 
        self.d_model = d_model   
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
    
    def forward(self, enc_input_ids, dec_input_ids, enc_padding_valid_len=None, dec_causal_mask_tensor=None):
        enc_embeddings = self.embedding(enc_input_ids)  # Output: (batch, seq_len, d_model)
        dec_embeddings = self.embedding(dec_input_ids)  # Output: (batch, seq_len, d_model)

        # 增加嵌入向量中元素的数值大小，防止词嵌入的数值范围远小于位置编码
        # 使得两者在相加时能够更均衡地贡献信息
        scale = math.sqrt(self.d_model) 
        enc_scaled_embeddings = enc_embeddings * scale
        dec_scaled_embeddings = dec_embeddings * scale
        
        enc_processed_input = self.pos_encoding(enc_scaled_embeddings)
        dec_processed_input = self.pos_encoding(dec_scaled_embeddings)
        
        enc_output = enc_processed_input
        for encoder_block in self.encoder:
            enc_output = encoder_block(enc_output, enc_padding_valid_len) 
            
        dec_output = dec_processed_input
        for decoder_block in self.decoder:
            dec_output = decoder_block(dec_output, enc_output, 
                                       cross_attention_valid_len=enc_padding_valid_len, 
                                       self_attention_causal_mask=dec_causal_mask_tensor)
            
        return self.linear(dec_output)