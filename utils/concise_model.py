import torchtext.vocab
import os
'''用于演示使用torchtext导入预训练的词嵌入模型'''

model_cache_dir = os.path.expanduser("~/.cache/torch/text/")
if not os.path.exists(model_cache_dir):
    os.makedirs(model_cache_dir,exist_ok=True)

# 加载 GloVe 模型，指定维度和缓存位置
# name='6B' 表示基于 6 billion tokens 训练
# dim=50 表示词向量维度为 50
glove = torchtext.vocab.GloVe(name='6B', dim=50, cache=model_cache_dir)

# 获取词向量
word = "king"
if word in glove.stoi:
    vec = glove[word]
    print(f"Vector for '{word}': {vec}")
    print(f"Shape of vector: {vec.shape}")
else:
    print(f"'{word}' not in GloVe vocabulary.")

# 对于不在词汇表中的词，GloVe 会返回一个 <unk> 向量
unknown_word = "supercalifragilisticexpialidocious"
unk_vec = glove[unknown_word]
print(f"Vector for unknown word '{unknown_word}': {unk_vec}")

# 查看词汇表大小和向量维度
print(f"GloVe vocabulary size: {len(glove.itos)}")
print(f"GloVe vector dimension: {glove.dim}")

# 获取所有向量的张量
# glove_vectors = glove.vectors
# print(f"Shape of all GloVe vectors: {glove_vectors.shape}")

# -----------------FastText和word2vec-----------------------

# 加载 FastText 词向量，并指定缓存目录
# print("尝试加载 FastText...")
# fasttext = torchtext.vocab.FastText(language='en', cache=custom_cache_dir)  # 英文 FastText

# 获取特定词的向量
# word = "example"
# if word in fasttext.stoi:
#     vec = fasttext[word]
#     print(f"FastText vector for '{word}': shape={vec.shape}\n")
# else:
#     print(f"'{word}' not in FastText vocabulary or OOV (FastText can generate for OOV)\n")


# 可以使用 Vectors 来加载预先下载的 Word2Vec 模型 (或其他自定义词向量)
# 注意: 'word2vec-google-news-300.vec' 是一个非常大的文件 。
# print("尝试加载 Vectors (e.g., Word2Vec)...")
# vectors_cache_dir = './vector_cache/' # 和你原始代码一致
# if not os.path.exists(vectors_cache_dir):
#     os.makedirs(vectors_cache_dir)
# vectors = torchtext.vocab.Vectors(name='word2vec-google-news-300.vec', # 你可能需要替换成一个小模型的文件名
#                                  cache=vectors_cache_dir)

# # 获取特定词的向量  
# word = "example"
# if word in vectors.stoi:
#     vec = vectors[word]
#     print(f"Word2Vec (via Vectors) vector for '{word}': shape={vec.shape}\n")
# else:
#     print(f"'{word}' not in Word2Vec (via Vectors) vocabulary\n")