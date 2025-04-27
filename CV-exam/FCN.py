import torch
import torch.nn as nn
import torchvision.models as models

class SimpleFCN(nn.Module):
    def __init__(self,num_classes) -> None:
        super().__init__()
        self.backbone = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.conv1x1=nn.Conv2d(512,num_classes,kernel_size=1)
        self.transpose_conv=nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,padding=16,stride=32)

    def forward(self,X):
        X=self.backbone(X)
        X=self.conv1x1(X)
        X=self.transpose_conv(X)
        return X
        
num_classes = 21  # 假设有21个类别
model = SimpleFCN(num_classes)

# 输入图像
input_image = torch.randn(1, 3, 224, 224)  # 假设输入图像尺寸为224x224

# 前向传播
output = model(input_image)
print(output.shape)