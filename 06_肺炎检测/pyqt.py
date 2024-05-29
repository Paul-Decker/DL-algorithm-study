import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt

import torch
from torchvision import models
from torch import nn
import numpy as np
from PIL import Image
from torch.nn import functional as F

import my_model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            # 把模型的所有层的参数的 requires_grad 属性设置为 false，表示不再对原本的参数进行训练
            param.requires_grad = False

# 初始化模型
def initialize_model(model_name, num_classes=3, feature_extract=False, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == 'myResnet':
        model_ft = my_model.ResNet()

        input_size = 512
    elif model_name == 'resnet34':
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                                    nn.LogSoftmax(dim=1))
        input_size = 224
    elif model_name == 'resnet101':
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                                    nn.LogSoftmax(dim=1))
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# 测试数据预处理（数据预处理的过程要与加载数据时的数据预处理过程保持一致）
def process_image(image_path, image_size=256, image_crop_size=224):
    # 读取测试数据
    img = Image.open(image_path)
    # 将图像对象 img 转换为RGB颜色模式（如果它原本不是这种颜色模式）
    img = img.convert('RGB')
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    # 如果图片宽度大于高度
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, image_size))
    else:
        img.thumbnail((image_size, 10000))
    # 如果不传入 image_crop_size 则表示不进行裁剪操作
    if image_crop_size is None:
        image_crop_size = image_size
    # Crop操作
    left_margin = (img.width-image_crop_size)/2
    bottom_margin = (img.height-image_crop_size)/2
    right_margin = left_margin + image_crop_size
    top_margin = bottom_margin + image_crop_size
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # 相同的预处理方法
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # 注意颜色通道应该放在第一个位置
    img = img.transpose((2, 0, 1))
    
    return img

class MyWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model = None
        self.input = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_ui(self):
        # 设置窗口标题
        self.setWindowTitle('肺炎诊断系统')
        self.setFixedSize(1200, 800)  # 设置窗口大小为固定值
        self.center()  # 将窗口显示在屏幕中央

        # 创建最外层布局器
        container = QVBoxLayout()

        # 程序名展示
        title = QLabel('肺炎诊断辅助')
        # 设置字体、字号、加粗
        title_font = QFont('Arial', 24)
        title.setFont(title_font)
        container.addWidget(title)
        # 添加固定间隔
        container.addSpacing(10)

        # 内层布局器
        layout_main = QHBoxLayout()
        container.addLayout(layout_main)

        # 展示区在内层布局器左侧
        layout_show = QVBoxLayout()
        layout_main.addLayout(layout_show)

        # 设置字体、字号、加粗
        font = QFont('Arial', 14)
        # 显示区域：图片显示
        self.image_label = QLabel('点击“上传图片”按钮上传待诊断胸部X光图片')
        self.image_label.setFont(font)
        self.image_label.setAlignment(Qt.AlignCenter)  # 设置图片居中显示
        self.image_label.setFixedSize(850, 550)
        # 显示区域：数字显示
        hbox = QHBoxLayout()
        # 创建三个 QLabel 用于显示数字
        self.label1 = QLabel('新冠肺炎：')
        self.label2 = QLabel('正常：')
        self.label3 = QLabel('普通肺炎：')
        self.label1.setFont(font)
        self.label2.setFont(font)
        self.label3.setFont(font)
        hbox.addWidget(self.label1)
        hbox.addWidget(self.label2)
        hbox.addWidget(self.label3)

        layout_show.addWidget(self.image_label)
        layout_show.addSpacing(20)
        layout_show.addLayout(hbox)

        # 按钮区在内层布局器右侧
        # 创建按钮
        btn_width, btn_height = 250, 70
        layout_button = QVBoxLayout()
        layout_main.addLayout(layout_button)
        self.upload_image_btn = QPushButton('上传图片')
        self.upload_image_btn.setFixedSize(btn_width, btn_height)  # 修改按钮大小
        self.choose_model_btn = QPushButton('选择模型（.pth文件）')
        self.choose_model_btn.setFixedSize(btn_width, btn_height)
        self.start_btn = QPushButton('开始诊断')
        self.start_btn.setFixedSize(btn_width, btn_height)
        # 给按钮绑定点击事件
        self.upload_image_btn.clicked.connect(self.open_image)
        self.choose_model_btn.clicked.connect(self.choose_model)
        self.start_btn.clicked.connect(self.display_posibility)
        # 设置显示区域，显示模型名称和模型准确率
        self.model_name = QLabel()
        self.model_acc = QLabel()
        self.model_name.setFixedSize(200, 25)    # 设置label固定大小
        self.model_acc.setFixedSize(200, 25)
        self.model_name.setFont(QFont('Arial', 10))
        self.model_acc.setFont(QFont('Arial', 10))
        layout_model = QVBoxLayout()
        layout_model.addWidget(self.model_name)
        layout_model.addWidget(self.model_acc)
        groupbox = QGroupBox("model information:")
        groupbox.setFixedSize(240, 120)
        groupbox.setLayout(layout_model)

        # 将按钮注册到“按钮”布局器中
        layout_button.addWidget(self.upload_image_btn)
        layout_button.addWidget(self.choose_model_btn)
        layout_button.addWidget(groupbox)
        layout_button.addWidget(self.start_btn)

        # 添加一个伸缩器
        container.addStretch()

        self.setLayout(container)

    def center(self):
        # 将窗口显示在屏幕中央
        window_rect = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        window_rect.moveCenter(center_point)
        self.move(window_rect.topLeft())

    # "上传图片"按钮点击事件
    def open_image(self):
        # 调用 QFileDialog.getOpenFileName 方法打开文件对话框，让用户选择一个图片文件
        image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        print(image_path)
        if image_path:
            # 使用 QPixmap 类加载选中的图片文件
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 使用 pixmap.scaled 方法将图片按比例缩放以适应 QLabel 的尺寸，保持图片的长宽比不变，防止图片变形
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), aspectRatioMode=True))

            # 数据预处理（数据预处理的过程要与加载数据时的数据预处理过程保持一致）
            img = process_image(image_path)
            img = torch.from_numpy(img)  # 将 numpy 数组转化为 tensor
            img = img.float()
            img = img.unsqueeze(0)  # 添加 batch 维度
            self.input = img.to(self.device)

            self.label1.setText(f'新冠肺炎：')
            self.label2.setText(f'正常：')
            self.label3.setText(f'普通肺炎：')
    
    # "选择模型"按钮点击事件
    def choose_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "choose model", "", "PTH Files (*.pth)")
        print(file_path)
        if file_path:
            model_name = file_path.split('_')[-3]
            self.model, self.input_size = initialize_model(model_name)
            self.model = self.model.to(self.device)

            # 加载 checkpoint 文件
            checkpoint = torch.load(file_path)
            # 加载训练好的参数
            self.model.load_state_dict(checkpoint['state_dict'])
            best_acc = checkpoint['best_acc']

            self.model_name.setText(f'model: {model_name}')
            self.model_acc.setText(f'accuracy: {best_acc*100:.3f}%')

    # "开始诊断"按钮点击事件
    def display_posibility(self):
        if self.model is None:
            QMessageBox.information(self, '提示', '还未选择模型，无法开始诊断')
        elif self.input is None:
            QMessageBox.information(self, '提示', '还未选择图片，无法开始诊断')
        else:
            self.model.eval()
            output = self.model(self.input)
            p = F.softmax(output, dim=1)
            p = p * 100
            p = p.cpu()
            # 在PyTorch中，detach() 方法用于从计算图中分离张量，返回一个新的张量，该张量不再具有梯度信息
            p = p.detach().numpy()
            p = p.flatten()
            p_COVID19, p_NORMAL, p_PNEUMONIA = p
            self.label1.setText(f'新冠肺炎：{p_COVID19:.3f}%')
            self.label2.setText(f'正常：{p_NORMAL:.3f}%')
            self.label3.setText(f'普通肺炎：{p_PNEUMONIA:.3f}%')

if __name__ == '__main__':
    # 创建程序
    app = QApplication(sys.argv)

    # 创建一个窗口
    w = MyWindow()
    w.show()

    # 程序进行循环等待状态
    app.exec()
