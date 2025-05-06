import torch.nn as nn

class Glove(nn.Module):
    '''glove实现'''
    def __init__(self,vocab_size,embed_dim):
        super().__init__()
        self.center_embed=nn.Embedding(vocab_size,embed_dim)
        self.context_embed=nn.Embedding(vocab_size,embed_dim)
        self.center_bias=nn.Embedding(vocab_size,1)
        self.context_bias=nn.Embedding(vocab_size,1)

    def forward(self,center_idx,context_idx):
        center_embed=self.center_embed(center_idx)
        context_embed=self.context_embed(context_idx)
        # bias 的形状是 (batch_size, 1)，需要 squeeze(1) 变成 (batch_size,)
        # 以便后续计算 loss 时可以直接加
        center_bias=self.center_bias(center_idx).squeeze(1)
        context_bias=self.context_bias(context_idx).squeeze(1)
        return center_embed,context_embed,center_bias,context_bias