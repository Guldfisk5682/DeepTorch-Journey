import torch
import torch.nn.functional as F
import torchtext.vocab as vocab 
import os
'''基于torchtext的Glove模型实现词相似性和词类比任务'''

def find_similar_tokens_glove(target_word, glove_vectors, top_n = 5) :
    """
    使用 torchtext.vocab.GloVe 加载的模型查找与给定单词语义最相似的 top_n 个单词。

    Args:
        target_word : 输入的单词。
        glove_vectors ): 已加载的 GloVe 模型实例。
        top_n : 返回最相似单词的数量。

    Returns:
        list[tuple[str, float]]: 包含 (相似词, 余弦相似度得分) 的列表。
    """
    if target_word not in glove_vectors.stoi:
        print(f"单词 '{target_word}' 不在 GloVe 词汇表中。")
        return []

    # 获取所有词向量和目标词的向量
    all_embeddings = glove_vectors.vectors
    target_word_idx = glove_vectors.stoi[target_word]
    # 获取词向量 (torchtext.vocab.GloVe.get_vecs_by_tokens 
    target_embedding = glove_vectors.get_vecs_by_tokens(target_word, lower_case_backup=True)
    if target_embedding.dim()==1:
        target_embedding=target_embedding.unsqueeze(0)

    # 计算余弦相似度
    all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
    target_embedding_norm = F.normalize(target_embedding, p=2, dim=1)
    cosine_similarities = torch.matmul(target_embedding_norm, all_embeddings_norm.transpose(0, 1)).squeeze(0)

    top_scores, top_indices = torch.topk(cosine_similarities, k=top_n + 1)

    similar_words_with_scores = []
    for score, idx in zip(top_scores, top_indices):
        idx_item = idx.item()
        # 跳过输入词本身
        if idx_item == target_word_idx:
            continue
        similar_words_with_scores.append((glove_vectors.itos[idx_item], score.item()))
    
    return similar_words_with_scores[:top_n]

def find_analogy_glove(word1, word2, word3, glove_vectors, top_n = 1) :
    """
    执行词类比任务，例如: word1 is to word2 as word3 is to ?
    计算公式: vec(word2) - vec(word1) + vec(word3)

    Args:
        word1 : 类比关系中的第一个词 (例如 "man")。
        word2 : 类比关系中的第二个词 (例如 "king")。
        word3 : 类比关系中的第三个词 (例如 "woman")。
        glove_vectors : 已加载的 GloVe 模型实例。
        top_n : 返回最相似的 top_n 个候选词。

    Returns:
        list[tuple[str, float]]: 包含 (候选词, 余弦相似度得分) 的列表。
    """
    input_words = [word1, word2, word3]
    for w in input_words:
        if w not in glove_vectors.stoi:
            print(f"单词 '{w}' 不在 GloVe 词汇表中。无法执行类比。")
            return []

    vec1 = glove_vectors.get_vecs_by_tokens(word1, lower_case_backup=True) 
    vec2 = glove_vectors.get_vecs_by_tokens(word2, lower_case_backup=True) 
    vec3 = glove_vectors.get_vecs_by_tokens(word3, lower_case_backup=True) 

    # 计算目标向量
    target_embedding = vec2 - vec1 + vec3
    if target_embedding.dim()==1:
        target_embedding=target_embedding.unsqueeze(0)
    

    # 获取所有词向量
    all_embeddings = glove_vectors.vectors # (vocab_size, dim)

    # # 计算余弦相似度
    all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
    target_embedding_norm = F.normalize(target_embedding, p=2, dim=1)
    cosine_similarities = torch.matmul(target_embedding_norm, all_embeddings_norm.transpose(0, 1)).squeeze(0)

    # 为了排除输入的三个词，我们可以获取足够多的候选，然后过滤
    # 或者在计算相似度后，将输入词的相似度分数设为一个非常小的值
    num_candidates_to_fetch = top_n + len(input_words) + 5 # 获取一些额外的候选
    top_scores, top_indices = torch.topk(cosine_similarities, k=num_candidates_to_fetch)

    analogy_results = []
    input_word_indices = {glove_vectors.stoi[w] for w in input_words}

    for score, idx in zip(top_scores, top_indices):
        idx_item = idx.item()
        # 跳过输入的词本身
        if idx_item in input_word_indices:
            continue
        analogy_results.append((glove_vectors.itos[idx_item], score.item()))
        if len(analogy_results) == top_n:
            break
            
    return analogy_results

if __name__ == '__main__':
    # --- 示例用法 ---
    model_cache_dir = os.path.expanduser("~/.cache/torch/text/")
    glove = vocab.GloVe(name='6B', dim=50,cache=model_cache_dir)

    target = "king"
    similar_words = find_similar_tokens_glove(target, glove, top_n=5)
    print(f"\n与 '{target}' 语义相似的词:")
    for word, score in similar_words:
        print(f"- {word}: {score:.4f}")

    target = "computer"
    similar_words = find_similar_tokens_glove(target, glove, top_n=5)
    print(f"\n与 '{target}' 语义相似的词:")
    for word, score in similar_words:
        print(f"- {word}: {score:.4f}")

    # 2. 测试词类比
    # man:king :: woman:? (期望 queen)
    w1, w2, w3 = "man", "king", "woman"
    analogies = find_analogy_glove(w1, w2, w3, glove, top_n=3)
    print(f"\n词类比: '{w1}' is to '{w2}' as '{w3}' is to ?")
    for word, score in analogies:
        print(f"- {word}: {score:.4f}")

    # france:paris :: germany:? (期望 berlin)
    w1, w2, w3 = "france", "paris", "germany"
    analogies = find_analogy_glove(w1, w2, w3, glove, top_n=3)
    print(f"\n词类比: '{w1}' is to '{w2}' as '{w3}' is to ?")
    for word, score in analogies:
        print(f"- {word}: {score:.4f}")
    
    # 测试不存在的词
    target = "supercalifragilisticexpialidocious" # 一个非常生僻的词
    print(f"\n测试不存在的词 '{target}':")
    similar_words = find_similar_tokens_glove(target, glove, top_n=5)
    if not similar_words:
        print(" (如预期，未找到相似词)")
