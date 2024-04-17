import torch 
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):

    '''
    ResNet Block
    '''
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.extra = nn.Sequential()
        if out_channel != in_channel:
            # [b, in_ch, h, w] => [b, out_ch, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        # x: [b, ch, h, w]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # short cut
        # x:[b, in_ch, h, w]  out:[b, out_ch, h, w]
        # in_ch 和 out_ch 不相同的时候，不能直接相加
        # extra(x)：[b, in_ch, h, w] => [b, out_ch, h, w]
        out = self.extra(x) + out

        return out

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # follow 4 blocks
        # [b, 64, h, w] => [b, 128, h, w]
        self.block1 = ResBlock(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.block2 = ResBlock(128, 256, stride=2)
        # [b, 256, h, w] => [b, 512, h, w]
        self.block3 = ResBlock(256, 512, stride=2)
        # [b, 512, h, w] => [b, 512, h, w]
        self.block4 = ResBlock(512, 512, stride=2)

        self.outlayer = nn.Linear(512*1*1, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # print('after conv:', x.shape) # [b, 512, 2, 2]

        # [b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool: ', x.shape)

        # 进入全连接层前，要进行 flatten 操作
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x

def main():
    block = ResBlock(64, 128, stride=4)
    temp = torch.randn(2, 64, 32, 32)
    out = block(temp)
    print('block', out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print('model:', out.shape)

    print(model)

if __name__ =='__main__':
    main()
