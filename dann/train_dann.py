from dann import DANN
import torch
import torch.optim as optim
import torch.nn as nn
import math
from torch.utils.data import Dataset,DataLoader

# 超参数设置
INPUT_DIM = 50       # 输入特征维度
HIDDEN_DIM = 100     # 隐藏层维度 
NUM_CLASSES = 2      # 类别数量 
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 0.01 # 学习率
ALPHA_GAMMA = 10     # 用于调度lambda值
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模拟数据生成
# 为了演示，我们生成简单的随机数据，并为它们添加域标签
# 源域数据 (X_s, Y_s): 有标签
# 目标域数据 (X_t): 无标签
def generate_dummy_data(num_samples, dim, num_classes, is_source=True):
    data = torch.randn(num_samples, dim) * 2
    if is_source:
        labels = torch.randint(0, num_classes, (num_samples,))
        return data, labels
    else:
        return data

NUM_SOURCE_SAMPLES = 1000
NUM_TARGET_SAMPLES = 1000

source_data, source_labels = generate_dummy_data(NUM_SOURCE_SAMPLES, INPUT_DIM, NUM_CLASSES, is_source=True)
target_data_unlabeled = generate_dummy_data(NUM_TARGET_SAMPLES, INPUT_DIM, NUM_CLASSES, is_source=False)

# 转换为 PyTorch Dataset 和 DataLoader (简化处理)
class DummyDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

source_dataset = DummyDataset(source_data, source_labels)
target_dataset = DummyDataset(target_data_unlabeled)

source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 实例化模型、损失函数和优化器
model = DANN(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
model.to(device)
criterion_label = nn.CrossEntropyLoss() # 用于标签预测 (Ly)
criterion_domain = nn.BCEWithLogitsLoss()   # 用于域分类 (Ld)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# 训练循环
print("开始训练 DANN 模型...")
for epoch in range(NUM_EPOCHS):
    # lambda (alpha) 的调度策略
    # lambda_p = 2 / (1 + exp(-gamma * p)) - 1
    # p 是训练进度，从 0 到 1
    p = float(epoch) / NUM_EPOCHS
    alpha = 2. / (1. + math.exp(-ALPHA_GAMMA * p)) - 1
    
    total_label_loss = 0.0
    total_domain_loss = 0.0
    total_samples = 0

    for i in range(len(source_loader)):
        for (source_data, source_labels),target_data in zip(source_loader,target_loader):
            source_data = source_data.to(device)
            source_labels = source_labels.to(device)
            target_data = target_data.to(device)

            # 合并源域和目标域数据，便于一次性处理特征提取
            combined_data = torch.cat((source_data, target_data), dim=0)

            # 为源域和目标域设置域标签
            # 源域标签为 0，目标域标签为 1
            domain_labels_source = torch.zeros(len(source_data), 1).to(device)
            domain_labels_target = torch.ones(len(target_data), 1).to(device)
            combined_domain_labels = torch.cat((domain_labels_source, domain_labels_target), dim=0).to(device)

            optimizer.zero_grad()

            # 前向传播
            label_preds, domain_preds = model(combined_data, alpha)

            # 计算标签预测损失 (仅针对源域数据)
            # 注意：label_preds 包含了源域和目标域的预测，需要切片
            loss_label = criterion_label(label_preds[:len(source_data)], source_labels)

            # 计算域分类损失 (对所有数据)
            loss_domain = criterion_domain(domain_preds, combined_domain_labels)

            # 总损失：标签损失 + 域分类损失
            # GRL 已经处理了域分类损失对特征提取器梯度的反转
            total_loss = loss_label + loss_domain

            # 反向传播
            total_loss.backward()
            optimizer.step()

            total_label_loss += loss_label.item() * len(source_data)
            total_domain_loss += loss_domain.item() * len(combined_data)
            total_samples += len(source_data) # 只统计有标签的源域样本作为进度依据

        avg_label_loss = total_label_loss / total_samples
        avg_domain_loss = total_domain_loss / (NUM_SOURCE_SAMPLES + NUM_TARGET_SAMPLES)
    if (epoch+1)%20==0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Alpha: {alpha:.4f}, Label Loss: {avg_label_loss:.4f}, Domain Loss: {avg_domain_loss:.4f}")

print("训练完成！")

model.eval()
with torch.no_grad():
    label_correct = 0
    label_total = 0
    for source_data, source_label in source_loader:
        source_data,source_label = source_data.to(device),source_label.to(device)
        label_preds, _ = model(source_data, alpha=0.0) # 评估时 GRL alpha 设为0，不影响特征
        _, predicted = torch.max(label_preds.data, 1)
        label_total += source_label.size(0)
        label_correct += (predicted == source_label).sum().item()
    print(f"\n源域标签分类精度: {100 * label_correct / label_total:.2f}%")

    # 理论上，训练好的 DANN 应该使得 domain_preds 接近 0.5 (无法区分)
    domain_correct = 0
    domain_total = 0
    
    # 评估源域的域分类
    for source_data, _ in source_loader:
        source_data = source_data.to(device)
        _, domain_preds = model(source_data, alpha=0.0)
        predicted_domain = (domain_preds > 0.5).float()
        domain_total += len(source_data)
        domain_correct += (predicted_domain == 0).sum().item() # 源域标签是0

    # 评估目标域的域分类
    for target_data in target_loader:
        target_data = target_data.to(device)
        _, domain_preds = model(target_data, alpha=0.0)
        predicted_domain = (domain_preds > 0.5).float()
        domain_total += len(target_data)
        domain_correct += (predicted_domain == 1).sum().item() # 目标域标签是1

    print(f"域分类器精度 (期望接近 50%): {100 * domain_correct / domain_total:.2f}%")