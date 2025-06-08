import torch
import math
import torch.nn as nn
from CBAM import CBAM
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self._get_kernel_size(channels, gamma, b),
                              padding=(self._get_kernel_size(channels, gamma, b) - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def _get_kernel_size(self, channels, gamma, b):
        k_size = int(abs((math.log(channels, 2) + b) / gamma))
        return k_size if k_size % 2 else k_size + 1  # 确保为奇数

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)  # (b, c, 1, 1)
        y = y.view(b, 1, c)  # (b, 1, c)
        y = self.conv(y)  # 一维卷积
        y = self.sigmoid(y)  # (b, 1, c)
        y = y.view(b, c, 1, 1)  # (b, c, 1, 1)
        return x * y.expand_as(x)  # 通道注意力加权

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, padding=0, stride=4, kernel_size=11),     # 卷积：(3, 224, 224)-->(96, 55, 55)
            nn.ReLU(inplace=True), # 激活函数
            #CBAM(96),  # 添加CBAM
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2),                   # 池化：(96, 27, 27)
            ECA(96)  # 添加ECA
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, padding=2, stride=1, kernel_size=5),  # 卷积：(96, 55, 55)-->(256, 27, 27)
            nn.ReLU(inplace=True),   # 激活函数
            CBAM(256),  # 添加CBAM
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2),                  # 池化：(256, 13, 13)
            ECA(256)  # 添加ECA
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, padding=1, stride=1, kernel_size=3),  # 卷积：(256, 27, 27)-->(384, 13, 13)
            nn.ReLU(inplace=True),                                               # 激活函数
            CBAM(384),  # 添加CBAM
            ECA(384)  # 添加ECA
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, padding=1, stride=1, kernel_size=3),  # 卷积：(384, 13, 13)-->(384, 13, 13)
            nn.ReLU(inplace=True),                                               # 激活函数
            CBAM(384),  # 添加CBAM
            ECA(384)  # 添加ECA
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, padding=1, stride=1, kernel_size=3),  # 卷积：(384, 13, 13)-->(256, 13, 13)
            nn.ReLU(inplace=True),                                              # 激活函数
            CBAM(256),  # 添加CBAM
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2),                    # 池化：(256, 13, 13)-->(256, 6, 6)
            #ECA(256)  # 添加ECA
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, padding=1, stride=1, kernel_size=3),
             #卷积：(384, 13, 13)-->(256, 13, 13)
            nn.ReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2),  # 池化：(256, 13, 13)-->(256, 6, 6)
            ECA(256)  # 添加ECA
        )

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4096, padding=0, stride=1, kernel_size=2),  # 全连接：(256, 6, 6)-->4096
            nn.ReLU(inplace=True),                                               # 激活函数
            nn.Dropout(p=0.7)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),                     # 全连接：4096-->4096
            nn.ReLU(inplace=True),                                               # 激活函数
            nn.Dropout(p=0.7)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=num_classes)              # 全连接：4096-->num_classes
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.fc1(x)
        x = torch.flatten(x, start_dim=1)  # 展平处理，start_dim=1:从channel的维度开始展开
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def _initialize_weights(self):      # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = AlexNet()
    print(model)
