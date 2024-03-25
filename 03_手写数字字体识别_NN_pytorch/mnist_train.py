import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt 

from utils import plot_image, plot_curve, one_hot


batch_size = 512

# step1. load dataset
# 加载训练集
train_loader = torch.utils.data.DataLoader(
    # 加载 MNIST 数据集，指定下载路径，指定数据是否为训练集
    # download=True表示如果当前目录没有训练集，则会在网上进行下载
    torchvision.datasets.MNIST('./03_手写数字字体识别_NN_pytorch/mnist_data/train', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                    # 将numpy格式转化为tensor格式
                                   torchvision.transforms.ToTensor(),
                                    # 正则化过程
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    # shuffle：洗牌，即随机打散数据集
    batch_size=batch_size, shuffle=True)

# 加载测试集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./03_手写数字字体识别_NN_pytorch/mnist_data/test', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

# 展示数据
x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')


# step2. 创建模型
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 线性层，xw+b
        # 第一个参数表示该层输入数据的个数，第二个参数表示输出数据的参数
        # 即第一个参数表示该层神经元个数，第二个参数表示该层的后一层的神经元个数
        self.fc1 = nn.Linear(28*28, 256)
        # 第二层的输入数据个数要与第一层的输出数据个数保持一致
        self.fc2 = nn.Linear(256, 64)
        # 第三层，也就是倒数第二层（最后一层应该是输出层）
        # 手写数字识别是个10分类任务，所以输出层应该是10个神经元
        self.fc3 = nn.Linear(64, 10)
    
    # 前向传播
    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # 一般来说，最后一层应该用softmax激活函数
        # 这里为了简单展示，先不加
        # h3 = h2w3+b3
        x = self.fc3(x)

        return x


# step3. 训练模型
net = Net()
num_iteration = 3
# net.parameters() 返回 [w1, b1, w2, b2, w3, b3] 各层的参数
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# 记录 loss 值，方便后续可视化处理
train_loss = []

for epoch in range(num_iteration):
    # 遍历数据集中的每个数据
    for batch_idx, (x, y) in enumerate(train_loader):
        # x: [b, 1, 28, 28]  y: [512]
        # [b, 1, 28, 28] => [b, feature]
        x = x.view(x.size(0), 28*28)
        # => [b, 10]
        out = net(x)
        # [b, 10]
        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)

        # 对之前的梯度进行清零，否则每次梯度计算结果会叠加
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 根据梯度更新参数 w 和 b ：w' = w - lr * grad
        optimizer.step()

        # 记录 loss 值
        train_loss.append(loss.item())

        # 每隔10个数据，显示一次 loss 值
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

plot_curve(train_loss)
# we get optimal [w1, b1, w2, b2, w3, b3]

# step4. 测试准确率
total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    # out: [b, 10] => pred: [b]
    pred = out.argmax(dim=1)
    # 预测正确的总数
    # .item()：从 tensor 中取出数值
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
# 准确度
acc = total_correct / total_num
print(f'test acc: {acc}')

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')