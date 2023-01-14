import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_1LeNet(nn.Module):
    # 重写构造函数
    def __init__(self):
        super(CNN_1LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 32x32 ——> 28x28
                in_channels=3,  # 输入通道数
                out_channels=16,  # 输出通道数
                kernel_size=5,  # 卷积核的大小
                stride=1,  # 步长
                padding=0  # 填充
            ),
            nn.ReLU(),
            nn.MaxPool2d(  # 28x28 ——> 14x14
                kernel_size=2,  # 卷积核的大小
                stride=2,  # 步长
                padding=0  # 填充
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(  # 14x14 ——> 10x10
                in_channels=16,  # 输入通道数
                out_channels=32,  # 输出通道数
                kernel_size=5,  # 卷积核的大小
                stride=1,  # 步长
                padding=0  # 填充
            ),
            nn.ReLU(),
            nn.MaxPool2d(  # 10x10 ——> 5x5
                kernel_size=2,  # 卷积核的大小
                stride=2,  # 步长
                padding=0  # 填充
            )
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 降维

            nn.Linear(32 * 5 * 5, 120),  # 全连接层
            nn.ReLU(),

            nn.Linear(120, 84),  # 全连接层
            nn.ReLU(),

            nn.Linear(84, 10)  # 全连接层
        )

    # 反向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        return x
