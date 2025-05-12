import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import random
from try_device import try_gpu
from tqdm import tqdm # 用于显示进度条
from BERT import BERTForPreTraining
from plot_loss_curver import LiveLossPlotter

# 配置参数
CONFIG={
    "dataset_name": "wikitext",
    "dataset_config_name": "wikitext-2-raw-v1", # 使用原始版本，未经过预处理
    "tokenizer_name": "bert-base-uncased", # BERT基础模型（不区分大小写）分词器
    "max_len": 128, 
    "batch_size": 16, 
    "num_epochs": 100, # 预训练通常需要很多epoch，这里为了演示设为100
    "learning_rate": 5e-5,
    "adam_epsilon": 1e-8,
    "warmup_steps": 0,
    "mlm_probability": 0.15, # 15% 的 token 会被mask
    "nsp_probability": 0.5, # 50% 的样本是 IsNext, 50% 是 NotNext

    # BERT模型参数 
    "vocab_size": None, # 将由tokenizer设置
    "num_encoder_layers": 6, # BERT-base通常是12，这里可以减少以加快演示
    "d_model": 768,      # BERT-base是768
    "ffn_hidden": 3072,  # d_model * 4
    "num_heads": 12,       # BERT-base是12
    "num_segments": 2,
    "dropout": 0.1,
    "layer_norm_eps": 1e-12
}

class WikiTextBertDataset(Dataset):
    '''
    处理抱抱脸datasets加载的WikiText数据
    转换为适合BERT NSP任务的成对句子
    '''
    def __init__(self,hf_dataset,tokenizer,max_len,nsp_prob):
        '''
        Args:
            hf_dataset: Hugging Face datasets库加载的原始数据集对象 
            tokenizer: Hugging Face tokenizer 对象
            max_len: BERT输入序列的最大长度
            nsp_prob: 生成NotNextSentence样本的概率 (50% IsNext, 50% NotNext)
        '''
        super().__init__()
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.nsp_prob=nsp_prob
        self.samples=[] # 存储 (tokens_a, tokens_b, is_next)
        
        # 数据预处理
        # 对于wikitext，需要提取'text'字段 文本按段落组织，段落之间由空行分隔
        # 段落内句子可能由换行分隔。会包含一些标题行 如 " = Gameplay = "
        all_lines=[]
        for example in tqdm(hf_dataset,desc="Processing datasets"):
            text=example['text'].strip() # 移除换行和空格
            if text and not text.startswith(" = ") and not text.startswith("= "):
                all_lines.append(text)
        # 构建NSP对
        num_lines=len(all_lines)
        for i in tqdm(range(num_lines-1),desc="Building NSP"):
            tokens_a=self.tokenizer.tokenize(all_lines[i])
            
            if random.random()<0.5:
                tokens_b=self.tokenizer.tokenize(all_lines[i+1])
                is_next=0 # 0代表is_next_sentence
            else:
                rand_idx=i
                while rand_idx==i or rand_idx==i+1:
                    if num_lines<=2:
                        if num_lines==1: # 只有一个句子无法构造NSP
                            tokens_b=tokens_a # 后续会忽略
                            break
                        elif i==0 and num_lines==2: # 只有两个句子
                            rand_idx=random.randrange(num_lines) # 这种情况为处理简单还是选择随机选择
                            break
                    rand_idx=random.randrange(num_lines)
                tokens_b=self.tokenizer.tokenize(all_lines[rand_idx])
                is_next=1 # 1代表not_next_sentence
            self.samples.append((tokens_a,tokens_b,is_next))
        print("Builded NSP samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        # 返回原始token列表和NSP标签 collate_fn会处理成BERT输入格式
        return self.samples[idx]
    
def bert_collate_fn(batch,tokenizer,max_len,mlm_prob):
    '''
    用于DataLoader,它负责将一批样本处理成BERT模型所需的格式
    负责：组合句子对 (A 和 B) 添加特殊token ([CLS], [SEP]) 进行MLM任务
    生成segment_ids, attention_mask, masked_lm_labels
    padding和max_len
    
    Args:
        batch: 一个列表，每个元素是 (tokens_a, tokens_b, is_next_label)
        tokenizer: Hugging Face tokenizer
        max_len: BERT输入序列的最大长度
        mlm_prob: Token被mask的概率
    '''
    
    batch_input_ids=[]
    batch_segment_ids=[]
    batch_attention_mask=[]
    batch_mlm_labels=[]
    batch_next_sentence_label=[]
    
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    
    for tokens_a,tokens_b,is_next_label in batch:
        # 组合AB句，添加cls和sep，同时确保总长度不超过max_len，优先保留A句
        truncate_seq_pair(tokens_a,tokens_b,max_len-3) # -3是为了给特殊符号留位置
        bert_tokens=[tokenizer.cls_token]+tokens_a+[tokenizer.sep_token]+tokens_b+[tokenizer.sep_token]
        input_ids=tokenizer.convert_tokens_to_ids(bert_tokens) # 将词元转换为对应idx
        
        # 创建seg_ids 0代表cls 第一sep和A句 1 代表B句和第二sep
        seg_ids=[0]*(len(tokens_a)+2)+[1]*(len(tokens_b)+1)
        
        # MLM任务，使用-100作为非掩码token，实际tokenid作为掩码token
        masked_input=list(input_ids) # 被部分遮蔽或随机替换的原始输入
        mlm_labels=[-100]*len(input_ids) # 记录的是未被遮蔽或者随机替换的目标序列
        
        # 不对cls和sep进行掩码
        cand_indices=[] # 存储的是除了特殊词元的对应idx
        for i,token_str in enumerate(bert_tokens):
            if token_str==tokenizer.cls_token or token_str==tokenizer.sep_token:
                continue
            cand_indices.append(i)
        
        random.shuffle(cand_indices) # 随机打乱候选mask位置
        num_to_mask=int(round(len(cand_indices)*mlm_prob)) # 需要mask的token数
        masked_token_count=0
        for index in cand_indices:
            if masked_token_count>=num_to_mask:
                break
            if random.random()<0.8: # 80%替换为[MASK]
                masked_token_id=mask_token_id
            else:
                if random.random()<0.5:
                    # 随机的一个词元id
                    masked_token_id=random.randint(0,tokenizer.vocab_size-1)
                else:
                    masked_token_id=input_ids[index] # 不变
            
            masked_input[index]=masked_token_id 
            mlm_labels[index]=input_ids[index]
            masked_token_count+=1

        # padding处理
        current_seq_len=len(masked_input)
        padding_len=max_len-current_seq_len
        input_ids_padded=masked_input+[pad_token_id]*padding_len
        seg_ids_padded=seg_ids+[0]*padding_len
        attention_mask_padded=[1]*current_seq_len+[0]*padding_len
        mlm_labels_padded=mlm_labels+[-100]*padding_len

        batch_input_ids.append(input_ids_padded)
        batch_segment_ids.append(seg_ids_padded)
        batch_attention_mask.append(attention_mask_padded)
        batch_mlm_labels.append(mlm_labels_padded)
        batch_next_sentence_label.append(is_next_label)

    return {
        "tokens_ids":torch.tensor(batch_input_ids,dtype=torch.long),
        "segment_ids":torch.tensor(batch_segment_ids,dtype=torch.long),
        "attention_mask":torch.tensor(batch_attention_mask,dtype=torch.long),
        "mlm_labels":torch.tensor(batch_mlm_labels,dtype=torch.long),
        "next_sentence_label":torch.tensor(batch_next_sentence_label,dtype=torch.long)
    }
        
def truncate_seq_pair(tokens_a,tokens_b,max_num_tokens):
    '''用于截断过长句子'''
    while True:
        total_length=len(tokens_a)+len(tokens_b)
        if total_length<=max_num_tokens:
            break
        if len(tokens_a)>len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            
def train_bert(model,dataloader,optimizer,scheduler,num_epochs,device):
    model.to(device)
    model.train()
    
    # 初始化绘图器
    plotter=LiveLossPlotter(xlabel="Epoch",ylabel="Loss")
    
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        total_loss=0
        progress_bar=tqdm(dataloader,desc="Training",leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            token_ids=batch["tokens_ids"].to(device)
            seg_ids=batch["segment_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            mlm_labels=batch["mlm_labels"].to(device)
            next_sentence_label=batch["next_sentence_label"].to(device)
            
            optimizer.zero_grad()
            loss,_,_=model(token_ids,seg_ids,attention_mask,
                mlm_labels,next_sentence_label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss+=loss.item()
            
            progress_bar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / (batch_idx + 1)})
        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        plotter.update(epoch+1,avg_epoch_loss)

    print("Training finished.")
    plotter.finalize(keep_open=True)
            

if __name__=="__main__":
    # 初始化tokenizer：
    # `AutoTokenizer.from_pretrained` 会自动下载并加载预训练tokenizer的配置和词汇表
    # 它能识别模型名称（如"bert-base-uncased"）并加载对应的tokenizer类（如BertTokenizerFast）
    print("加载tokenizer")
    tokenizer=AutoTokenizer.from_pretrained(CONFIG["tokenizer_name"])
    CONFIG["vocab_size"] = tokenizer.vocab_size # 更新CONFIG中的vocab_size
    print(f"Vocab size: {CONFIG['vocab_size']}")
    # 特殊词元
    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"MASK token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    
    # 加载和处理训练集
    print("加载数据集")
    # wikitext-2-raw-v1 只有 train, validation, test 三个split
    raw_datasets=load_dataset(CONFIG["dataset_name"],CONFIG["dataset_config_name"],)
    train_dataset_raw = raw_datasets['train'].select(range(6000)) # 只取前6000条做测试
    print(f"training instance : {train_dataset_raw[0]}")
    
    bert_train_dataset = WikiTextBertDataset(
        hf_dataset=train_dataset_raw,
        tokenizer=tokenizer,
        max_len=CONFIG["max_len"],
        nsp_prob=CONFIG["nsp_probability"]
    )
    
    # 创建DataLoader
    print("创建DataLoader")
    train_dataloader=DataLoader(
        bert_train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True, # 训练时打乱数据
        collate_fn=lambda batch: 
            bert_collate_fn(batch, tokenizer, CONFIG["max_len"], CONFIG["mlm_probability"])
    )
    print(f"Num batches: {len(train_dataloader)}")
    
    # 创建BERT模型
    print("初始化BERT模型")
    model = BERTForPreTraining(
        vocab_size=CONFIG["vocab_size"],
        num_encoder=CONFIG["num_encoder_layers"],
        d_model=CONFIG["d_model"],
        ffn_hidden=CONFIG["ffn_hidden"],
        num_heads=CONFIG["num_heads"],
        max_len=CONFIG["max_len"], # BERT Embedding中的max_len
        num_segments=CONFIG["num_segments"],
        dropout=CONFIG["dropout"],
        layer_norm_eps=CONFIG["layer_norm_eps"]
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params/1e6:.2f} M")
    
    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], eps=CONFIG["adam_epsilon"])
    # get_linear_schedule_with_warmup 在warmup步内，学习率从0线性增加到初始lr 接着线性衰减到0
    num_training_steps = len(train_dataloader) * CONFIG["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=CONFIG["warmup_steps"],
                                                num_training_steps=num_training_steps)
    print("优化器和调度器设置完成")
    
    # 训练模型
    device=try_gpu()
    print(f"device : {device}")
    
    print("开始训练...")
    train_bert(model,train_dataloader,optimizer,scheduler,CONFIG["num_epochs"],device)
    print("训练完成")

    # print("保存模型中")
    # torch.save(model.state_dict(),"../model/BERT_model.pth")
    # print("保存完成")