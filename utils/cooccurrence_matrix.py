import torch
def build_cooccurrence_matrix(tokenized_sentences, vocab, window_size, weighted = True):
    '''
    构建共现矩阵
    Args:
        tokenized_sentences (List[List[str]]): 分词后的句子列表.
        vocab (Vocabulary): 词汇表实例.
        window_size (int): 上下文窗口大小 (单侧).
        weighted (bool): 是否使用距离倒数加权. Defaults to True.

    Returns:
        torch.Tensor: 共现矩阵 (vocab_size x vocab_size).
    '''
    vocab_size = len(vocab)
    unk_idx = vocab.unk
    pad_idx = vocab.token2idx.get(vocab.pad, -1) # 如果没有pad则设为-1

    matrix = torch.zeros(vocab_size, vocab_size, dtype=torch.float32)

    for sentence in tokenized_sentences:
        indices = vocab[sentence]
        sentence_len = len(indices)
        for j, center_idx in enumerate(indices): # 遍历中心词索引 j 和值 center_idx
            # 跳过 <unk> 和 <pad> 作为中心词
            if center_idx == unk_idx or center_idx == pad_idx:
                continue
            start_idx = max(0, j - window_size)
            end_idx = min(sentence_len, j + window_size + 1)
            
            for i in range(start_idx, end_idx): # i是上下文词索引
                if i == j:
                    continue
                context_idx = indices[i]
                if context_idx == unk_idx or context_idx == pad_idx:
                    continue
                if weighted:
                    distance = abs(j - i)
                    weight = 1.0 / distance
                else:
                    weight = 1.0
                matrix[center_idx, context_idx] += weight
    return matrix
