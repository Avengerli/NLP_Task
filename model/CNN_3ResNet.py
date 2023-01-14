import torch.nn as nn
import torch.nn.functional as F


class CNN_3ResNet(nn.Module):
    # 重写构造函数
    def __init__(self, block):
        super(CNN_3ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(                      # 224x224 ——> 112x112
                in_channels=3,              # 输入通道数
                out_channels=64,            # 输出通道数
                kernel_size=7,              # 卷积核的大小
                stride=2,                   # 步长
                padding=3,                  # 填充
                bias=False                  # 取消偏执
            ),
            nn.BatchNorm2d(64),             # 批归一化
            nn.ReLU(inplace=True),
            nn.MaxPool2d(                   # 112x112 ——> 56x56
                kernel_size=3,              # 卷积核的大小
                stride=2,                   # 步长
                padding=1                   # 填充
            )
        )
        self.in_channels = 64
        self.layer1 = self.make_layer(block,  64, 2, 1)     # 56x56 ——> 28x28
        self.layer2 = self.make_layer(block, 128, 2, 2)     # 28x28 ——> 14x14
        self.layer3 = self.make_layer(block, 256, 2, 2)     # 14x14 ——> 7x7
        self.layer4 = self.make_layer(block, 512, 2, 2)     # 7x7   ——> 1x1
        self.classifier = nn.Sequential(
            #
            nn.AdaptiveAvgPool2d((1, 1)),   # 池化
            #
            nn.Flatten(),                   # 降维
            #
            nn.Linear(512, 100)             # 全连接层
        )

    # 加载ResBlock
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    # 反向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x


class ResBlock(nn.Module):
    # 重写构造函数
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,        # 输入通道数
                out_channels=out_channels,      # 输出通道数
                kernel_size=3,                  # 卷积核的大小
                stride=stride,                  # 步长
                padding=1,                      # 填充
                bias=False                      # 偏执
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,       # 输入通道数
                out_channels=out_channels,      # 输出通道数
                kernel_size=3,                  # 卷积核的大小
                stride=1,                       # 步长
                padding=1,                      # 填充
                bias=False                      # 偏执
            ),
            nn.BatchNorm2d(out_channels)
        )
        # 如果步长不是1，或者通道数发送变化
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,    # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    kernel_size=1,              # 卷积核的大小
                    stride=stride,              # 步长
                    padding=0,                  # 填充
                    bias=False                  # 偏执
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    # 反向传播
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        out = F.relu(out)
        return out
