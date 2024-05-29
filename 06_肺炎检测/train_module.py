''' 
训练模块
'''

import os
import time
import torch
import copy
import pickle

def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=1, directory='checkpoint', filename='checkpoint.pth'):
    # 检查当前工作目录中是否存在指定的目录
    if not os.path.exists(directory):
        # 如果不存在，则创建目录
        os.makedirs(directory)
        print(f"目录 '{directory}' 创建成功！")
    else:
        print(f"目录 '{directory}' 已存在。")
    filename = directory + '/' + filename
    
    since = time.time()
    best_acc = 0
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    # 是否用GPU训练
    if not torch.cuda.is_available():
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    # 用于后续训练过程的保存
    train_history = {
        'val_acc_history': val_acc_history,
        'train_acc_history': train_acc_history,
        'valid_losses': valid_losses, 
        'train_losses': train_losses,
        'LRs': LRs,
        'epoch': num_epochs
    } 

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练   
            else:
                model.eval()   # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):         
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # 取出概率最大的
                    _, preds = torch.max(outputs, 1)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                  'state_dict': model.state_dict(),     # 模型的所有参数
                  'best_acc': best_acc,
                  'optimizer' : optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc.to('cpu').numpy())
                valid_losses.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc.to('cpu').numpy())
                train_losses.append(epoch_loss)
        
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        train_history['epoch'] = epoch+1
        # 每隔5个epoch就保存一次训练过程（防止中途终止程序时丢失训练过程）
        if epoch%5 == 0:
            # 将字典保存到文件中
            history_filename = directory + '/train_history.pkl'
            with open(history_filename, 'wb') as file:
                pickle.dump(train_history, file)
    
    # 完整训练好后，保存最后一次记录
    history_filename = directory + '/train_history.pkl'
    with open(history_filename, 'wb') as file:
        pickle.dump(train_history, file)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, train_history