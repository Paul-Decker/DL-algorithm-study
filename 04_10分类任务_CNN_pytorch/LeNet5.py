import torch 
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    
    def __init__(self):
        super(Lenet5, self).__init__()

        # 卷积层单元
        self.conv_unit = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            # 第二层
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            # 输入数据 x: [b, 3, 32, 32] 经过两层卷积后=> [b, 16, 5, 5]

        )
        # flatten
        # fc unit 全连接层单元
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        # use Cross Entropy Loss（对于分类问题，一般用交叉熵）
        # self.criteon = nn.CrossEntropyLoss()


    def forward(self, x):
        batch_size = x.size(0)
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)
        # flatten（展开成一维）: [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batch_size, -1)
        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)

        return logits




def main():
    net = Lenet5()
    
    # test
    temp = torch.randn(2, 3, 32, 32)
    out = net(temp)
    print(out.shape)

if __name__ == '__main__':
    main()