import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from LeNet5 import Lenet5
from ResNet import ResNet18

def main():
    batch_size = 32
    
    cifar_train = datasets.CIFAR10('./04_10分类任务_CNN_pytorch/cifar', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10('./04_10分类任务_CNN_pytorch/cifar', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    # for x, label in iter(cifar_train):
    #     print('x:', x.shape, 'label:', label.shape)
    #     break

    device = torch.device('cuda')
    # model = Lenet5()
    model = ResNet18()
    model.to(device)
    
    # 打印网络模型的数据结构
    # print(model)

    # 包含了 softmax 操作
    criteon = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        # 将模型设置为训练模式
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # x: [b, 3, 32, 32]
            # label: [b]
            x, label = x.to(device), label.to(device)
            logits = model(x)
            # logits: [b ,10]
            # label: [b]
            # loss: tensor scalar （一维tensor）
            loss = criteon(logits, label)
             # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, loss.item())

        # 将模型设置为测试模式
        model.eval()
        # with torch.no_grad() 包住的代码块不会计算梯度，即不会影响 model
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # x: [b, 3, 32, 32]
                # label: [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)

                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor（一维tensor）
                # 统计这个batch中预测正确的数量
                total_correct += torch.eq(pred, label).float().sum()
                total_num += x.size(0)
            # 正确率
            acc = total_correct / total_num
            print(epoch, acc)
        

if __name__ == '__main__':
    main()