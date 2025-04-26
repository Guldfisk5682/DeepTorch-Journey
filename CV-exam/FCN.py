import torch
import torch.nn as nn
import torchvision.models as models

class SimpleFCN(nn.Module):
    def __init__(self,num_classes) -> None:
        super().__init__()
        self.backbone=models.resnet18(pretrained=True).children()[:-2]
        