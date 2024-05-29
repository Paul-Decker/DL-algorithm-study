import os
import torchvision
from matplotlib import pyplot as plt 


# 检查哪些层需要训练
def print_need_train_layer(model):
    params_to_update = model.parameters()
    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

# 展示模型效果：绘制 train loss、valid loss、train acc、valid acc、learning rate 关于 epoch 的变化
def show_model_performace(train_history, directory):
    num_epochs = train_history['epoch']
    # 检查当前工作目录中是否存在指定的目录
    if not os.path.exists(directory):
        # 如果不存在，则创建目录
        os.makedirs(directory)
        print(f"目录 '{directory}' 创建成功！")
    else:
        print(f"目录 '{directory}' 已存在。")

    # 绘制损失曲线
    plt.plot(train_history['train_losses'], label='Train Loss')
    plt.plot(train_history['valid_losses'], label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()  # 添加图例
    # plt.xticks(range(0, num_epochs, 1))  # 设置 x 轴间隔为 1
    plt.grid(True)  # 显示网格
    plt.savefig(directory + '/loss_image.png')
    plt.show()

    # 绘制训练和验证准确率曲线
    plt.plot(train_history['train_acc_history'], label='Train Accuracy')
    plt.plot(train_history['val_acc_history'], label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    # plt.xticks(range(0, num_epochs, 1))  # 设置 x 轴间隔为 1
    plt.grid(True)  # 显示网格
    plt.savefig(directory + '/acc_image.png')
    plt.show()

    # 绘制学习率变化曲线
    plt.plot(train_history['LRs'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.savefig(directory + '/LRs.png')
    plt.show()