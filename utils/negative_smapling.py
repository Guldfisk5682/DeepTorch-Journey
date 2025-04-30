from skip_gram import SkipGramModel
from Vocab import Vocabulary
from Vocab import tokenize
from apply_word_embed import find_similar_token
from Vocab import subsampling
import torch
import torch.nn.functional as F
import torch.optim as optim 
import pandas as pd
from datasets import load_dataset

def negative_sampling(center_words,vocab_size,n_sample):
    '''
    Args:
        center_words (torch.Tensor): 中心词的索引，形状为 (batch_size,)
        vocab_size (int): 词汇表的大小
        n_samples (int): 每个中心词生成的负样本数量
    '''
    batch_size=center_words.size(0)
    device=center_words.device
    sampling_weights=torch.ones(vocab_size,device=device)
    neg_sample=torch.multinomial(
        sampling_weights,
        batch_size*n_sample,
        replacement=True
    ).reshape(batch_size,n_sample) # 返回的形状是(batch_size,n_sample)
    
    while True:
        # 确保负样本不包含中心词
        mask=(neg_sample==center_words.unsqueeze(1))
        if not mask.any():
            break
        new_samples=torch.multinomial(
            sampling_weights,
            mask.sum().item(),
            replacement=True
        )
        neg_sample[mask]=new_samples
    return neg_sample

def loss_function(center_embed,context_embed,neg_embed):
    '''
    计算负采样的损失函数。

    Args:
        center_embeds (torch.Tensor): 中心词的嵌入，形状为 (batch_size, embed_dim)
        context_embeds (torch.Tensor): 上下文词的嵌入，形状为 (batch_size, embed_dim)
        neg_embeds (torch.Tensor): 负样本的嵌入，形状为 (batch_size, n_samples, embed_dim)
    '''
    batch_size,_=center_embed.shape
    
    # 通过向量点积的思想衡量向量相似度(方向越近越相似)
    pos_score=torch.sum(center_embed*context_embed,dim=1)
    # 对正例，我们希望 sigmoid(pos_score) → 1
    # 希望最大化 log(sigmoid(pos_score))，这会推动 pos_score 变大，即正例对的词向量更相似
    pos_loss=F.logsigmoid(pos_score).sum()
    
    neg_score=torch.bmm(neg_embed,center_embed.unsqueeze(2)).squeeze()
    # 负例，我们希望 sigmoid(neg_score) → 0，等价于希望 sigmoid(-neg_score) → 1
    # 希望最大化 log(sigmoid(-neg_score))，这会推动 neg_score 变小，即负例对的词向量不相似
    neg_loss=F.logsigmoid(-neg_score).sum()
    
    # 由上述，我们希望最大化得到的两个loss,但一般来说
    # 优化算法是最小化损失函数，因此添加一个负号希望该值趋于0
    total_loss=-(pos_loss+neg_loss)/batch_size
    return total_loss



def get_skipgram_data(sentences,window_size):
    '''
    为每个中心词创建一个上下文窗口
    Args:
        sentences (List[List[str]]): 词元化后的句子列表
        window_size (int): 上下文窗口的大小
    '''
    center_words = []
    context_words = []
    
    for sentence in sentences:
        for idx,center_word in enumerate(sentence):
            for i in range(-window_size,window_size+1):
                context_idx=idx+i
                if context_idx<0 or context_idx>=len(sentence) or context_idx==idx:
                    continue
                context_word=sentence[context_idx]
                # 训练样本是(中心词,上下文词)
                # 每一个样本都需要记录中心词
                center_words.append(center_word)
                context_words.append(context_word)
                
    return center_words,context_words

def train_skipgram(model,sentences,vocab,window_size,n_samples,lr,epochs,device):
    '''
    训练跳元模型
     Args:
        model (SkipGramModel): Skip-gram 模型实例
        sentences (List[List[str]]): 词元化后的句子列表
        vocab (Vocabulary): 词汇表实例
        window_size (int): 上下文窗口的大小
        n_samples (int): 每个中心词的负样本数量
        learning_rate (float): 学习率
        epochs (int): 训练轮数
        device (str): 'cpu' 或 'cuda'
    '''
    model.to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    
    for epoch in range(epochs):
        center_words,context_words=get_skipgram_data(sentences,window_size)
        # 转换为索引
        center_indices=torch.tensor(vocab[center_words],dtype=torch.long).to(device)
        context_indices=torch.tensor(vocab[context_words],dtype=torch.long).to(device)
        
        batch_size=16
        num_batch=len(center_indices)//batch_size
        total_loss=0
        
        for i in range(num_batch):
            start=i*batch_size
            end=(i+1)*batch_size
            center_batch = center_indices[start:end]
            context_batch = context_indices[start:end]
            
            # 负采样
            neg_samples = negative_sampling(center_batch, len(vocab), n_samples)
            neg_samples = neg_samples.to(device)
            
            # 向前传播
            center_embeds = model.forward_center(center_batch)
            context_embeds = model.forward_context(context_batch)
            neg_embeds = model.forward_context(neg_samples)
            
            loss = loss_function(center_embeds, context_embeds, neg_embeds)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batch:.4f}")
        
    
if __name__ == '__main__':
    # 准备数据
    # raw_dataset=load_dataset('ag_news',cache_dir="../data")
    # train_set=raw_dataset['train']
    # train_set.to_csv("../data/ag_news_train.csv",index=False)
    train_set=pd.read_csv("../data/ag_news_train.csv")
    train_set=train_set['text'].head(3000).tolist() # 选取前3000行作为训练数据
    tokenized_sentences=[tokenize(text) for text in train_set]
    
    # 构建词汇表
    vocab = Vocabulary(tokens=tokenized_sentences, freq_threshold=3,t=1e-4)
    tokenized_sentences=subsampling(tokenized_sentences,vocab)
    # 创建模型
    embed_dim = 100
    model = SkipGramModel(len(vocab), embed_dim)

    # 训练参数
    window_size = 2
    n_samples = 5
    learning_rate = 0.001
    epochs = 20 # 数据仅用作示例
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 训练模型
    train_skipgram(model, tokenized_sentences, vocab, window_size, n_samples, learning_rate, epochs, device)
    torch.save(model,'../model/skip_gram.pth')
    # similar_token=find_similar_token('and',model,vocab,2)