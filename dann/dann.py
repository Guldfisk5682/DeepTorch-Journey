import torch.nn as nn
from grl import GRL

class DANN(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_classes):
        super(DANN, self).__init__()
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        # 标签分类器
        self.label_predictor = nn.Linear(hidden_dim, num_classes)

        # 域分类器
        self.domain_predictor=nn.Linear(hidden_dim,1)
        
        # 梯度反转层
        self.grl = GRL()
    
    def forward(self, x,alpha=1.0):
        # 设置alpha(lambda)值
        self.grl.set_alpha(alpha)
        
        # 特征提取
        features = self.feature_extractor(x)
        
        # 标签预测
        label_output = self.label_predictor(features)
        
        # 将特征传递给 GRL，然后传递给域分类器
        # 这里 reversed_features 是 GRL 前向传播的结果，也就是 features 本身
        # 但是在反向传播时，来自 domain_classifier 的梯度会通过 GRL 被翻转
        reversed_features = self.grl(features)
        domain_output = self.domain_predictor(reversed_features)
        
        return label_output, domain_output