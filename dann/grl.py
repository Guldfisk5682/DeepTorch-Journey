import torch.nn as nn
from torch.autograd import Function

class GradientReversalFn(Function):
    """
    梯度反转层：
    前向传播时，输入不变。
    反向传播时，梯度乘以 -alpha。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        # 保存 alpha 到 context，以便在 backward 中使用
        ctx.alpha = alpha
        # 前向传播时，直接返回输入 x
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时，将梯度乘以 -alpha
        # 注意：如果 forward 方法有多个输入，backward 方法也需要返回对应数量的梯度
        # 这里 x 是第一个输入，alpha 是第二个输入。alpha 是一个标量，不需要梯度，所以返回 None
        return -ctx.alpha * grad_output, None

# 创建一个方便调用的 GRL 模块
class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

    def set_alpha(self, alpha):
        self.alpha = alpha