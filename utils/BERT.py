import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import EncoderBlock

class BERTEmbedding(nn.Module):
    '''
    BERT输入表示层
    整合了词元、片段和位置嵌入
    输入数据说明：
    - 词元ID序列每个元素是词汇表中的一个 ID，会包含 [CLS] 和 [SEP] token 的 ID
        已经接受过<pad>填充和截断操作
    - 片段ID序列用于区分输入序列中哪些词属于哪些句子 (句子 A 的词标记为 0，句子 B 的词标记为 1)
      - 对于单句输入，所有 token 的 segment_id 通常都是 0
      - 属于第一个片段（包括 [CLS] 和第一个 [SEP]）的所有 token 的 segment_id 通常是 0
      - 属于第二个片段（包括第二个 [SEP]）的所有 token 的 segment_id 通常是 1
    segment_ids 的长度必须与 token_ids 的长度 seq_len 完全相同
    '''
    def __init__(self,vocab_size,d_model,max_len=512,num_segments=2,dropout=0.5):
        '''
        num_segments允许模型为预定义的几种片段类型学习不同的基础嵌入。这些基础嵌入为后续的
        Transformer 层提供了一个关于片段身份的信号，
        帮助模型在处理整个序列时更好地区分和理解不同片段的内容和它们之间的关系。
        '''
        super().__init__()
        self.token_embedding=nn.Embedding(vocab_size,d_model) # 词元嵌入
        self.segment_embedding=nn.Embedding(num_segments,d_model) # 片段嵌入
        # BERT的PE层是可学习的 不是像原始 Transformer 那样使用固定的 sin/cos 函数
        self.position_embedding=nn.Embedding(max_len, d_model)# 位置嵌入
        self.layer_norm=nn.LayerNorm(d_model,eps=1e-12) # BERT常用epsilon值
        self.dropout=nn.Dropout(dropout)
        self.d_model=d_model
        
    def forward(self,token_ids,segment_ids):
        # token_ids: (batch_size, seq_len)
        # segment_ids: (batch_size, seq_len)
        # output_shape:(batch_size, seq_len, d_model)
        token_embed=self.token_embedding(token_ids)
        seg_embed=self.segment_embedding(segment_ids)
        seq_len=token_ids.size(1)
        pos_ids=torch.arange(seq_len,dtype=torch.long,device=token_ids.device)
        pos_ids=pos_ids.unsqueeze(0).expand_as(token_ids)
        pos_embed=self.position_embedding(pos_ids)
        
        combined_embedding=pos_embed+token_embed+seg_embed
        embedding=self.dropout(self.layer_norm(combined_embedding))
        return embedding
        
class BERTMLMHead(nn.Module):
    '''
    MLM遮蔽语言模型
    接收来自 BERT 主体（Transformer Encoders）的输出，并为被遮蔽的词元生成预测。
    '''
    def __init__(self,vocab_size,d_model,layer_norm_eps=1e-12,token_embedding_weight=None):
        super().__init__()
        # ----转换层-----
        # 包含一个全连接层，GELU激活，和LayerNorm
        # 提供了一些额外的非线性变换能力
        self.transform_dense=nn.Linear(d_model,d_model)
        self.transform_activation=nn.GELU()
        self.transform_layer_norm=nn.LayerNorm(d_model,eps=layer_norm_eps)
        
        # ----解码器层----
        # 这个线性层将 d_model 维的表示映射到词汇表大小的 logits
        # 全连接层的权重可以与 BERT 的词嵌入层共享权重
        # 通常 bias=False 或有一个单独的偏置参数
        self.decoder=nn.Linear(d_model,vocab_size,bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size)) # 单独的偏置项
        if token_embedding_weight is not None:
            self.decoder.weight=token_embedding_weight
        
    def forward(self,sequence_output):
        '''
        sequence_output:来自BERT最后一层Transformer Encoder的输出，
        形状是(batch_size, seq_len, d_model)
        MLM任务通常使用交叉熵损失，因此期望的输入是原始、未经
        softmax/logsigmoid的logits
        '''
        transformed_output = self.transform_dense(sequence_output)
        transformed_output = self.transform_activation(transformed_output)
        transformed_output = self.transform_layer_norm(transformed_output)

        # 应用解码器层得到 logits
        # 注意：如果 self.decoder.weight 来自词嵌入，它通常是 (vocab_size, d_model)
        # nn.Linear 内部会自动处理转置，所以直接调用即可
        prediction_logits = self.decoder(transformed_output) + self.decoder_bias
        return prediction_logits
    
class BERTNSPHead(nn.Module):
    '''NSP下一句预测'''
    def __init__(self,d_model):
        super().__init__()
        #----Pooler层----
        # 该层可选 输入是CLS token的最终隐藏状态向量
        # Pooler 层的作用就是对[CLS] token 的表示进行进一步的变换和“池化”，
        # 使其更适合用作整个序列的代表性特征。
        # 对 [CLS] 的表示进行进一步的非线性变换，
        # 使其更适合用于下游的分类任务。可以看作是对序列表示的一种“池化”或精炼。
        # self.pooler=nn.Linear(d_model,d_model)
        # self.activation=nn.Tanh()
        # ----Classifier层----
        # 输入是Pooler的输出或CLS的表示
        # 输出维度2代表IsNext或NotNext
        self.classifer=nn.Linear(d_model,2)
        
    def forward(self,pooled_output):
        '''
        pooled_output 来自 BERT 最后一层 Transformer Encoder 
        的[CLS] token对应的输出向量 shape:(batch_size, d_model)
        BERT主体输出是(batch,seq_len,d_model)
        需要从这个张量中提取出 [CLS] token 的部分，即 sequence_output[:, 0, :]
        '''
        return self.classifer(pooled_output)
    
class BERTEncoder(nn.Module):
    '''BERT编码器'''
    def __init__(self,vocab_size,num_encoder,d_model,ffn_hidden,
                 num_heads,max_len,num_segments,dropout=0.5,):
        super().__init__()
        self.bertembedding=BERTEmbedding(vocab_size,d_model,max_len,num_segments,dropout)
        self.encoderList=nn.ModuleList(
            [EncoderBlock(d_model,ffn_hidden,num_heads,dropout) for _ in range(num_encoder)] )
        # Pooler层
        self.pooler_dense=nn.Linear(d_model,d_model)
        self.pooler_activation=nn.Tanh()
        
    def forward(self,token_ids,seg_ids,attention_mask):
        '''
        这里的attention_mask通常是(batch,seq_len),1表示真实token，0表示padding
        传递给EncoderBlock的自注意力层前需要特殊处理,padding不参与注意力运算
        先前设计的Encoder模块中，期望的max_len掩码是(batch,)或(batch,1)
        '''
        valid_lens = attention_mask.sum(dim=1).to(device=token_ids.device) # (batch_size,)
                    
        encoder_output = self.bertembedding(token_ids, seg_ids)
        for encoder in self.encoderList:
            encoder_output=encoder(encoder_output,valid_lens)
        sequence_output=encoder_output
        cls_token = sequence_output[:, 0, :] # 提取 [CLS] token 
        pooled_output = self.pooler_activation(self.pooler_dense(cls_token))
        return sequence_output, pooled_output
    
class BERTForPreTraining(nn.Module):
    def __init__(self,vocab_size,num_encoder,d_model,ffn_hidden,
                 num_heads,max_len,num_segments,dropout=0.5,
                 layer_norm_eps=1e-12):
        super().__init__()
        self.bert=BERTEncoder(vocab_size,num_encoder,d_model,ffn_hidden,
                                 num_heads,max_len,num_segments,dropout)
        self.nsp_head=BERTNSPHead(d_model)
        token_embedding_weight=self.bert.bertembedding.token_embedding.weight
        self.mlm_head=BERTMLMHead(vocab_size,d_model,layer_norm_eps,token_embedding_weight)
    
    def forward(self,token_ids,seg_ids,attention_mask,
                masked_lm_labels=None, next_sentence_label=None):
        sequence_output,pooled_output=self.bert(token_ids,seg_ids,attention_mask)
        mlm_logits=self.mlm_head(sequence_output)
        nsp_logits=self.nsp_head(pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_func=nn.CrossEntropyLoss()
            # 计算MLM损失 mlm_logits: (batch_size, seq_len, vocab_size)
            # masked_lm_labels: (batch_size, seq_len), 
            # 其中非遮蔽位置通常用一个特殊值（如 -100）标记以忽略
            mlm_loss=loss_func(mlm_logits.reshape(-1,mlm_logits.shape[-1]),
                               masked_lm_labels.reshape(-1))
            # 计算NSP损失
            # nsp_logits: (batch_size, 2)
            # next_sentence_label: (batch_size,)
            nsp_loss=loss_func(nsp_logits,next_sentence_label)
            total_loss=mlm_loss+nsp_loss
            return total_loss,mlm_logits,nsp_logits
        else:    
            return mlm_logits,nsp_logits