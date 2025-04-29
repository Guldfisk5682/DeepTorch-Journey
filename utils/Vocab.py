import re
from collections import Counter
'''
用到的正则符号解释
\d: 匹配数字 +： 匹配一次或多次 |： 逻辑或，表示匹配前或后的字符
\w: 匹配任意大小写字母，数字和下划线 
(?:...)：?:放在括号内开头表示非捕获组，意思是regex会匹配?:后的内容
但是不会将这部分单独捕获或存储，只关心整个模式匹配到的完整字符串
*： 匹配零次或多次
'''

def tokenize(text:str):
    '''简单的词元化函数'''
    if text is None:
        return []
    pattern=r"\d+\.\d+|\w+(?:'\w+)*|\d+"
    # NLP分词一般将文本转换为小写
    tokens=re.findall(pattern,text.lower())
    return tokens

class Vocabulary:
    '''
    处理文本数据的词汇表类
    构建词到索引和索引到词的映射
    处理特殊标记<unk>,<pad>,<eos>,<bos>
    过滤低频词
    tokens是列表或嵌套列表，是tokenize后的文本数据
    '''
    def __init__(self,tokens=None,freq_threshold=1,pad="<pad>",unk="<unk>",\
        eos="<eos>",bos="<bos>"):
        self._unk=unk
        self.pad=pad
        self.eos=eos
        self.bos=bos
        if tokens is None:
            tokens=[]
        if tokens and isinstance(tokens[0],list):
            tokens=[token for sublist in tokens for token in sublist]
        counter=Counter(tokens)
        counter=sorted(counter.items(),key=lambda x:x[1],reverse=True)
        vocab_tokens=[unk,pad,bos,eos]
        tag=set(vocab_tokens)
        vocab_tokens.extend(token for token,idx in counter if \
        idx>=freq_threshold and token not in tag)
        self.token2idx={token:idx for idx,token in enumerate(vocab_tokens)}
        self.idx2token={idx:token for idx,token in enumerate(vocab_tokens)}

        
    def __getitem__(self,token):
        '''根据token获取索引'''
        if not isinstance(token,(tuple,list)):
            # 未知词返回unk
            return self.token2idx.get(token,self.unk)
        else:
            return [self.token2idx.get(t,self.unk) for t in token]
        
    def __len__(self):
        return len(self.token2idx)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    @property
    def unk(self):
        return self.token2idx[self._unk]
    
    @staticmethod
    def collate_batch(batch_texts,vocab,tokenize,max_len=None,add_bos=False,add_eos=False):
        '''
        批处理文本数据，进行词元化和填充
        Args:
            原始批量文本句子
            实例化词汇表对象
            词元化函数
            最大序列长度
        '''
        processed_seq=[]
        lengths=[]
        bos_idx=vocab[vocab.bos]
        eos_idx=vocab[vocab.eos]    
        for text in batch_texts:
            tokens=tokenize(text)
            indices=vocab[tokens]
            if add_bos:
                indices.insert(0,bos_idx)
            if add_eos:
                indices.append(eos_idx)
            processed_seq.append(indices)
        
        if max_len is not None:
            processed_seq=[seq[:max_len] for seq in processed_seq]
        
        lengths=[len(seq) for seq in processed_seq]
        max_length=max(lengths) if lengths else 0
        pad=vocab[vocab.pad]
        padded_sequences=[]
        for seq in processed_seq:
            padded_len=max_length-len(seq)
            padded_seq=seq+[pad]*padded_len
            padded_sequences.append(padded_seq)
        
        return padded_sequences,lengths
# -------------------------------

corpus = [
    "This is the first sentence.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
all_tokens = [token for text in corpus for token in tokenize(text)]
vocab_instance = Vocabulary(all_tokens, freq_threshold=1)

batch = [
    "This is a short sequence.",
    "This is a slightly longer sequence to test padding."
]

# 调用静态方法
padded_batch, batch_lengths = Vocabulary.collate_batch(
    batch_texts=batch,
    vocab=vocab_instance,
    tokenize=tokenize,
    add_eos=True
)

for seq, length in zip(padded_batch, batch_lengths):
    print(f"原始长度 (含EOS): {length}, 填充后序列: {seq}")