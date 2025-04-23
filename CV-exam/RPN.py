import torch
from torchvision.models.detection.rpn import \
RegionProposalNetwork, AnchorGenerator, RPNHead
from torchvision.models.detection.image_list import ImageList

# 假设我们有一个特征图
feature_map = torch.rand(1, 3, 32, 32)

# 定义锚框生成器，使用嵌套的元组
anchor = AnchorGenerator(
    sizes=((16, 32, 64),),  # 每个特征图的大小
    aspect_ratios=((0.5, 1, 2),)  # 每个特征图的宽高比
)

# 定义RPN头部网络
rpn_head = RPNHead(3, 9)

# 初始化RPN
rpn = RegionProposalNetwork(
    anchor,
    rpn_head,
    fg_iou_thresh=0.7,
    bg_iou_thresh=0.3,
    batch_size_per_image=256,
    positive_fraction=0.5,
    nms_thresh=0.7,
    pre_nms_top_n=dict(training=1000, testing=1000),
    post_nms_top_n=dict(training=1000, testing=1000),
)

# 将模型设置为评估模式
rpn.eval()

features = {'0': feature_map}
image_list=ImageList(feature_map,[(32,32)])

# 期望Images是一个ImageList对象
out = rpn(image_list, features)  # 将 features 作为字典传递

print(out)