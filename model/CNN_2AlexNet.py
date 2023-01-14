import torch.nn as nn


class CNN_2AlexNet(nn.Module):
    # 重写构造函数
    def __init__(self):
        super(CNN_2AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(                      # 227x227 ——> 55x55
                in_channels=3,              # 输入通道数
                out_channels=96,            # 输出通道数
                kernel_size=11,             # 卷积核的大小
                stride=4,                   # 步长
                padding=0                   # 填充
            ),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(                   # 55x55 ——> 27x27
                kernel_size=3,              # 卷积核的大小
                stride=2,                   # 步长
                padding=0                   # 填充
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(                      # 27x27 ——> 27x27
                in_channels=96,             # 输入通道数
                out_channels=256,           # 输出通道数
                kernel_size=5,              # 卷积核的大小
                stride=1,                   # 步长
                padding=2                   # 填充
            ),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(                   # 27x27 ——> 13x13
                kernel_size=3,              # 卷积核的大小
                stride=2,                   # 步长
                padding=0                   # 填充
            )
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(                      # 13x13 ——> 13x13
                in_channels=256,            # 输入通道数
                out_channels=384,           # 输出通道数
                kernel_size=3,              # 卷积核的大小
                stride=1,                   # 步长
                padding=1                   # 填充
            ),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(                      # 13x13 ——> 13x13
                in_channels=384,            # 输入通道数
                out_channels=384,           # 输出通道数
                kernel_size=3,              # 卷积核的大小
                stride=1,                   # 步长
                padding=1                   # 填充
            ),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(                      # 13x13 ——> 13x13
                in_channels=384,            # 输入通道数
                out_channels=256,           # 输出通道数
                kernel_size=3,              # 卷积核的大小
                stride=1,                   # 步长
                padding=1                   # 填充
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(                   # 13x13 ——> 6x6
                kernel_size=3,              # 卷积核的大小
                stride=2,                   # 步长
                padding=0                   # 填充
            )
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # 降维

            nn.Linear(256 * 6 * 6, 4096),   # 全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),          # 全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 100)            # 全连接层
        )

    # 反向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        return x
