import torch
import torchvision.ops as ops

# 假设我们有一个特征图，形状为 (batch_size, channels, height, width)
feature_map = torch.rand(1, 256, 32, 32)  # 1 个样本，256 个通道，32x32 分辨率

# 定义感兴趣区域的边界框，形状为 (num_boxes, 5)
# 每个边界框的格式为 (batch_index, x1, y1, x2, y2)
boxes = torch.tensor([
    [0, 10, 10, 20, 20],  # 第 0 个样本，边界框坐标为 (10, 10, 20, 20)
    [0, 5, 5, 15, 15],    # 第 0 个样本，边界框坐标为 (5, 5, 15, 15)
]).float()

# 定义输出大小
output_size = (7, 7)  # 池化后的特征图大小为 7x7

# 定义空间缩放比例
spatial_scale = 1.0  # 假设特征图的大小与输入图像相同

# 使用 roi_pool 进行池化
pooled_features = ops.roi_pool(feature_map, boxes, output_size, spatial_scale)

# 输出结果
print(pooled_features.shape)  # 输出: torch.Size([2, 256, 7, 7])