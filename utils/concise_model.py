import torchtext

glove = torchtext.vocab.GloVe(name='6B', dim=100)  # 也可以选择 dim=50, 200, 300

# 获取特定词的向量
word = "example"
if word in glove.stoi:  # stoi: string to index
    vec = glove[word]
    print(f"GloVe vector for '{word}': shape={vec.shape}")
else:
    print(f"'{word}' not in vocabulary")
    

# 加载 FastText 词向量
fasttext = torchtext.vocab.FastText(language='en')  # 英文 FastText

# 获取特定词的向量
word = "example"
if word in fasttext.stoi:
    vec = fasttext[word]
    print(f"FastText vector for '{word}': shape={vec.shape}")
else:
    print(f"'{word}' not in vocabulary")
    

# 可以使用 Vectors 来加载预先下载的 Word2Vec 模型
vectors = torchtext.vocab.Vectors(name='word2vec-google-news-300.vec', 
                                 cache='./vector_cache/')

# 获取特定词的向量  
word = "example"
if word in vectors.stoi:
    vec = vectors[word]
    print(f"Word2Vec vector for '{word}': shape={vec.shape}")
else:
    print(f"'{word}' not in vocabulary")