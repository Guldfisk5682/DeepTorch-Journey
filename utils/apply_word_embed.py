import torch
import torch.nn.functional as F

def find_similar_token(word,model,vocab,top_n=1,mode='center'):
    '''
     查找与给定单词语义最相似的 top_n 个单词。

    Args:
        word (str): 输入的单词。
        model (SkipGramModel): 训练好的 SkipGram 模型。
        vocab (Vocabulary): 对应的词汇表。
        top_n (int): 返回最相似单词的数量。
        mode (str): 使用 'center' (in_embeddings) 还是 'context' (out_embeddings)。
                               默认为 'center'，通常效果更好。

    '''
    model.eval()
    # device = next(model.parameters()).device

    word_idx=vocab[word]
    if word_idx==vocab.unk:
        print("the word cannot be found")
        return []
    
    if mode=='center':
        embeddings = model.in_embeddings.weight.detach() # (vocab_size, embed_dim)
    elif mode == 'context':
        embeddings = model.out_embeddings.weight.detach()

    input_embedding = embeddings[word_idx].unsqueeze(0) # (1, embed_dim)

    # 计算余弦相似度,使用L2归一化
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    input_embedding_norm = F.normalize(input_embedding, p=2, dim=1)

    cosine_similarities = torch.matmul(input_embedding_norm,\
    embeddings_norm.transpose(0, 1)).squeeze(0) # (vocab_size,)

    top_results=torch.topk(cosine_similarities,k=top_n+1) 
    # 返回前k个最大值以及在cosine中的索引
    similar_word=[]
    for score,idx in zip(top_results.values,top_results.indices):
        if idx.item()==word_idx:
            continue
        similar_word.append((vocab.idx2token[idx.item()],score.item()))
    return similar_word[:top_n]