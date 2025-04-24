import torch
from torchvision.models.detection.rpn import \
RegionProposalNetwork, AnchorGenerator, RPNHead
from torchvision.models.detection.image_list import ImageList

# 假设我们有一个特征图，格式为 (batch_size, channels, height, width)
feature_map = torch.rand(1, 3, 32, 32)

# 定义锚框生成器，使用嵌套的元组
anchor = AnchorGenerator(
    sizes=((16, 32, 64),),  # 每个特征图的大小
    aspect_ratios=((0.5, 1, 2),)  # 每个特征图的宽高比
)

# 定义RPN头部网络,输入通道数为3，锚框数量为9(sizes*aspect_ratios)
rpn_head = RPNHead(3, 9)

# 初始化RPN
rpn = RegionProposalNetwork(
    anchor,
    rpn_head,
    fg_iou_thresh=0.7,  # 前景IoU阈值
    bg_iou_thresh=0.3,  # 背景IoU阈值
    batch_size_per_image=256,  # 每张图像的样本数
    positive_fraction=0.5,  # 正样本比例
    nms_thresh=0.7,  # 非极大值抑制阈值
    pre_nms_top_n=dict(training=2000, testing=2000),  # 训练和测试时的前N个候选框
    post_nms_top_n=dict(training=1000, testing=1000),  # 训练和测试时的后N个候选框
)

# 将模型设置为评估模式
rpn.eval()

# 将特征图放入字典中，键为特征图的层名
# 这里表示的就是第0层特征图,也就是输入的特征图(CNN提取的特征图)
features = {'0': feature_map}

# ImageList会自动将传入的特征图和图像大小匹配 
# 图像大小是一个包含图像高宽的元组列表 
image_list = ImageList(feature_map, [(32, 32)])

# 需求的输入:
# 特征图格式为(batch_size, channels, height, width)
# ImageList对象,包含特征图和图像的大小信息
out = rpn(image_list, features)

# 输出结果
print(out)