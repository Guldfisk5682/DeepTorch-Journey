import torch.nn as nn

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        # 中心词嵌入层
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        # 上下文词嵌入层
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward_center(self, center_indices):
        """获取中心词的嵌入"""
        return self.in_embeddings(center_indices)

    def forward_context(self, context_indices):
        """获取上下文/负采样词的嵌入(用于计算分数)"""
        return self.out_embeddings(context_indices)
    
