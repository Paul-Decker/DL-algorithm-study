'''
加载数据
'''

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

data_dir = './data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# data_transforms 指定了所有图像预处理操作
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),#从中心开始裁剪
        #transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        # transforms.RandomRotation(5),#随机旋转，-5到5度之间随机选
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    'valid': transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 8

# 数据集：{train: ..., valid: ...}
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']
}
# dataloaders['train']：包含 train 集的所有数据 + 每个数据对应的 label
#       可以理解为数组 [num_batch, [batch_size, channel, w, h], [batch_size]]
# num_batch 表示 batch 的个数（即 num_batch = 数据量/batch_size）
'''
num_batch = 100
batch_size = 8
即train数据集总数据量为800张图片则 
for x, label in dataloaders['train']:    
    print(x.shape, label.shape)

循环 100 遍，打印：[8, c, w, h] [8]    8张图片一组+8个label
'''


# 数据集的数据个数：{train: ..., valid: ...}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
# 各个分类的名字
class_names = image_datasets['train'].classes