import torch
from torch.utils.data import TensorDataset, DataLoader
from try_device import try_gpu

def glove_loss_fn(center_indices, context_indices, x_values, glove_model, x_max=100, alpha=0.75):
    '''
    glove的损失函数
    Args:
        center_indices: 一批中心词索引
        context_indices: 一批上下文词索引
        x_values: 对应的共现值 X_ij
        glove_model: glove模型
        x_max: 权重函数截断阈值
        alpha: 权重函数指数
    '''
    device = next(glove_model.parameters()).device
    center_indices = center_indices.to(device)
    context_indices = context_indices.to(device)
    x_values = x_values.to(device)

    center_embed, context_embed, center_bias, context_bias = glove_model(center_indices, context_indices)

    # 计算预测的对数共现概率
    log_x = torch.log(x_values + 1e-8) 
    # 执行逐元素乘法
    predicted = torch.sum(center_embed * context_embed, dim=1) + center_bias + context_bias

    # 权重函数,为了不过分强调出现次数极高的词
    weights = torch.where(x_values < x_max, (x_values / x_max) ** alpha, torch.ones_like(x_values, dtype=torch.float32))
    weights = weights.to(device) 

    # 计算加权平方误差损失
    loss = torch.mean(weights * (predicted - log_x) ** 2) 
    return loss

def train_glove(model, cooccurrence_matrix, epochs, lr=0.01, train_batch_size=16, x_max=100, alpha=0.75):
    '''
    训练glove模型
    Args:
        model: glove模型
        cooccurrence_matrix: 共现矩阵
        epochs: 训练轮次
        learning_rate: 学习率
        train_batch_size: 训练批处理大小
        x_max: 权重函数截断阈值 (for loss_function)
        alpha: 权重函数指数 (for loss_function)
    '''
    device=try_gpu()
    model.to(device)
    # Glove原始论文选择使用Adagrad
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    # 确保不为0
    non_zero_indices = torch.nonzero(cooccurrence_matrix.to(device), as_tuple=False)

    center_word_indices = non_zero_indices[:, 0]
    context_word_indices = non_zero_indices[:, 1]
    x_values = cooccurrence_matrix[center_word_indices, context_word_indices].float()

    dataset = TensorDataset(center_word_indices, context_word_indices, x_values)
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=0) 

    for epoch in range(epochs):
        total_loss=0
        for batch_center, batch_context, batch_x_ij in dataloader:
            loss = glove_loss_fn(batch_center, batch_context, batch_x_ij, model, x_max=x_max, alpha=alpha)
            total_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        