# !/usr/bin/env python3


import datetime
import glob
import os
import sys
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

debug = False

embed_size = 128        # 嵌入层的大小（即词向量的维度）
hidden_size = 1024      # 隐藏层的大小（隐藏状态的维度）
lr = 0.001              
lstm_layers = 2
batch_size = 32
epochs = 15
seq_len = 48

if debug:
    batch_size = 2
    epochs = 1000


class LyricsDataset(Dataset):
    def __init__(self, seq_len, file="data/lyrics.txt"):
        SOS = 0  # start of song
        EOS = 1  # end of song

        self.seq_len = seq_len
        with open(file, encoding="utf-8") as f:
            lines = f.read().splitlines()

        # 创建一个字典word2index，用于将单词映射到它们的索引
        self.word2index = {"<SOS>": SOS, "<EOS>": EOS}

        # Convert words to indices
        indices = []
        num_words = 0
        for line in lines:
            indices.append(SOS)
            for word in line:
                if word not in self.word2index:
                    self.word2index[word] = num_words
                    num_words += 1
                indices.append(self.word2index[word])
            indices.append(EOS)

        # 创建一个index2word字典，用于将索引映射回它们的单词
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.data = np.array(indices, dtype=np.int64)

    def __len__(self):
        # 样本总数（句子的总个数） = 文字总数 / 序列长度
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, i):
        start = i * self.seq_len
        end = start + self.seq_len
        return (
            torch.as_tensor(self.data[start:end]),  # input
            torch.as_tensor(self.data[start + 1 : end + 1]),  # output
        )


class LyricsNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, lstm_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, lstm_layers, batch_first=True)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)

    def forward(self, word_ids, lstm_hidden=None):
        # Embed word ids into vectors
        embedded = self.embedding(word_ids)

        # Forward propagate LSTM
        lstm_out, lstm_hidden = self.lstm(embedded, lstm_hidden)

        # Forward propagate linear layer
        out = self.h2h(lstm_out)

        # Decode hidden states to one-hot encoded words
        out = self.h2o(out)

        return out, lstm_hidden


def accuracy(output, target):
    """Compute the accuracy between model output and ground truth.

    Args:
        output: (batch_size, seq_len, vocab_size)
        target: (batch_size, seq_len)

    Returns:
        float: accuracy value between 0 and 1
    """
    output = output.reshape(-1, vocab_size)
    target = target.flatten()

    a = output.topk(1).indices.flatten()
    b = target
    return a.eq(b).sum().item() / len(a)


def generate(start_phrases):
    # Convert to a list of start words.
    # i.e. '宁可/无法' => ['宁可', '无法']
    start_phrases = start_phrases.split("/")

    hidden = None

    def next_word(input_word):
        nonlocal hidden     # 使用nonlocal关键字，表示该变量不是该函数的局部变量

        input_word_index = dataset.word2index[input_word]
        input_ = torch.Tensor([[input_word_index]]).long().to(device)
        output, hidden = model(input_, hidden)
        top_word_index = output[0].topk(1).indices.item()
        return dataset.index2word[top_word_index]

    result = []  # a list of output words
    cur_word = "/"

    for i in range(seq_len):
        if cur_word == "/":  # end of a sentence
            result.append(cur_word)
            next_word(cur_word)     # 这一步改变了hidden（隐藏状态）

            if len(start_phrases) == 0:
                break

            for w in start_phrases.pop(0):
                result.append(w)
                cur_word = next_word(w)

        else:
            result.append(cur_word)
            cur_word = next_word(cur_word)

    # Convert a list of generated words to a string 将列表转化为字符串
    result = "".join(result)
    result = result.strip("/")  # remove trailing slashes 删除首尾的斜杠
    return result


def training_step():
    # 每一次for循环训练一个batch，整个for循环结束则表示完成一次epoch的训练
    for i, (input_, target) in enumerate(train_loader):
        model.train()

        input_, target = input_.to(device), target.to(device)

        output, _ = model(input_)
        loss = F.cross_entropy(output.reshape(-1, vocab_size), target.flatten())

        # Make sure gradient does not accumulate 清除之前的梯度，以防止梯度累积
        optimizer.zero_grad()  
        # Compute gradient 计算梯度
        loss.backward()  
        # Update NN weights 更新参数
        optimizer.step()  

        acc = accuracy(output, target)

        print(
            "Training: Epoch=%d, Batch=%d/%d, Loss=%.4f, Accuracy=%.4f"
            % (epoch, i, len(train_loader), loss.item(), acc)
        )

        if not debug:
            step = epoch * len(train_loader) + i
            # 使用writer（TensorBoard的摘要写入器）记录训练损失和准确率
            writer.add_scalar("loss/training", loss.item(), step)
            writer.add_scalar("accuracy/training", acc, step)

            if i % 50 == 0:
                generated_lyrics = generate("深/度/学/习")
                writer.add_text("generated_lyrics", generated_lyrics, i)
                writer.flush()


def evaluation_step():
    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for data in test_loader:
            input_, target = data[0].to(device), data[1].to(device)

            output, _ = model(input_)
            loss = F.cross_entropy(output.reshape(-1, vocab_size), target.flatten())

            epoch_acc += accuracy(output, target)

            epoch_loss += loss.item()
    
    # 计算整个验证集的平均损失和准确率
    epoch_loss /= len(test_loader)
    epoch_acc /= len(test_loader)
    print(
        "Validation: Epoch=%d, Loss=%.4f, Accuracy=%.4f"
        % (epoch, epoch_loss, epoch_acc)
    )

    if not debug:
        writer.add_scalar("loss/validation", epoch_loss, epoch)
        writer.add_scalar("accuracy/validation", epoch_acc, epoch)
        writer.flush()


def save_checkpoint():
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "checkpoint-%s.pth" % datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
    )


def load_checkpoint(file):
    global epoch

    # torch.load()函数会反序列化文件中的内容，并返回一个包含模型状态信息的字典
    ckpt = torch.load(file)

    print("Loading checkpoint from %s." % file)

    # 使用模型的load_state_dict方法加载检查点中保存的模型参数
    model.load_state_dict(ckpt["model_state_dict"])

    # 使用优化器的load_state_dict方法加载检查点中保存的优化器参数
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = ckpt["epoch"]


if __name__ == "__main__":
    # Create cuda device to train model on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define dataset
    dataset = LyricsDataset(seq_len=seq_len)

    # Split dataset into training and validation
    data_length = len(dataset) # 样本总数（句子总数）
    lengths = [int(data_length - 1000), 1000]
    train_data, test_data = random_split(dataset, lengths)

    # Create data loader
    '''
    shuffle=True：
    这个参数指定在每个epoch（即一次遍历整个数据集）开始时，DataLoader 应该随机打乱数据集的顺序。
    这样做可以帮助模型更好地学习，因为它不会总是看到数据集中的相同顺序的样本。
    '''
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    if debug:
        train_loader = [next(iter(train_loader))]
        test_loader = [next(iter(test_loader))]

    # Sanity check（健全性检测）: view training data
    if False:
        i = 0
        for data in train_loader:
            if i >= 10:
                break
            input_batch, _ = data
            first_sample = input_batch[0]

            pprint("".join([dataset.index2word[x.item()] for x in first_sample]))
            i += 1

    # Create NN model
    vocab_size = len(dataset.word2index)
    model = LyricsNet(
        vocab_size=vocab_size,      # 单词总数
        embed_size=embed_size,      # embedding层的维度（词向量的维度）
        hidden_size=hidden_size,    # 隐藏层的维度
        lstm_layers=lstm_layers,    # lstm的层数
    )
    model = model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Load checkpoint
    ''' 
    glob.glob 是一个内置函数，用于从文件系统获取与特定模式匹配的文件列表
    '''
    checkpoint_files = glob.glob("checkpoint-*.pth")
    if (
        not debug
        and len(checkpoint_files) > 0
        and input("Enter y to load %s: " % checkpoint_files[-1]) == "y"
    ):
        load_checkpoint(checkpoint_files[-1])
    else:
        epoch = 0

    # 如果用户输入'y'，则运行歌词生成器
    if (
        input("Enter y to enter inference mode, anything else to enter training mode: ")
        == "y"
    ):
        # Inference loop
        while True:
            start_words = input("Enter start-words divided by '/' (e.g. '深/度/学/习'): ")
            if not start_words:
                break

            print(generate(start_words))
    # 否则开始训练模型
    else:
        if not debug:
            '''
            SummaryWriter 是 torch.utils.tensorboard 模块中的一个类，
            它提供了一个高级接口来与TensorBoard交互。TensorBoard是一个可视化工具，
            由Google开发，用于监控机器学习实验。它可以帮助用户理解、调试和优化训练过程。 

            SummaryWriter 对象负责将数据记录到一个日志目录中，这个目录可以被TensorBoard读取和展示。
            '''
            writer = SummaryWriter()

        # Optimization loop
        while epoch < epochs:
            training_step()
            evaluation_step()

            if not debug:
                save_checkpoint()

            epoch += 1
