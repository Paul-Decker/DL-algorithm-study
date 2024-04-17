import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
import math

from multilyer_perceptron import MultilayerPerception

data = pd.read_csv('./data/mnist-demo.csv')
numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
# 创建一个新画布（指定大小为10*10（单位：英寸））
plt.figure(figsize=(10, 10))
for plot_index in range(numbers_to_display):
    # 图像的像素值（data是个二维数组，这里的digit取到了二维数组中第plot_index这一行，并用.values转为numpy数组）
    digit = data[plot_index:plot_index+1].values  # digit是个1*785的数组
    digit_label = digit[0][0]  # digit的第一个数据是标签（即“正确答案”）
    digit_pixels = digit[0][1:]  # digit_pixels是个一维数组 除去第一个后，剩下的数据就是所有像素点（28*28=784个像素点，即输入层的784个神经元）
    # 图像大小（28*28）
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    frame = digit_pixels.reshape((image_size, image_size))
    plt.subplot(num_cells, num_cells, plot_index+1)
    # 通过 imshow() 函数在 Matplotlib 查看器上直接根据像素点绘制出相应的图像
    plt.imshow(frame, cmap='Greys')
    plt.title(digit_label)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# pandas中的DataFrame.sample方法：对DataFrame进行简单随机抽样，frac参数接收一个float类型数据，指定随机抽取行或列的比例
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

# DataFrame转为numpy数组
train_data = train_data.values
test_data = test_data.values

num_training_examples = 5000

x_train = train_data[:num_training_examples, 1:]
y_train = train_data[:num_training_examples, [0]]

x_test = test_data[:, 1:]
y_test = test_data[:, [0]]

layers = [784, 25, 10]

normalize_data = True
# 迭代次数，即损失值减少的次数或权重参数更新的次数
max_iterations = 500
# 学习率
alpha = 0.1

multilayer_perception = MultilayerPerception(x_train, y_train, layers, normalize_data)
(thetas, costs) = multilayer_perception.train(max_iterations, alpha)
plt.plot(range(len(costs)), costs)
plt.xlabel('Gradient steps')
plt.ylabel('costs')
plt.show()

y_train_predictions = multilayer_perception.predict(x_train)
y_test_predictions = multilayer_perception.predict(x_test)

train_p = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
test_p = np.sum(y_test_predictions == y_test) / y_test.shape[0] * 100

print('训练集准确率：', train_p)
print('测试集准确率：', test_p)

# 可视化展示
numbers_to_display = 64
num_cells = math.ceil(math.sqrt(numbers_to_display))
# 创建一个新画布（指定大小为10*10（单位：英寸））
plt.figure(figsize=(15, 15))
for plot_index in range(numbers_to_display):
    digit_label = y_test[plot_index, 0]
    digit_pixels = x_test[plot_index, :]

    predicted_label = y_test_predictions[plot_index][0]

    image_size = int(math.sqrt(digit_pixels.shape[0]))
    frame = digit_pixels.reshape((image_size, image_size))

    color_map = 'Greens' if predicted_label == digit_label else 'Reds'
    plt.subplot(num_cells, num_cells, plot_index+1)
    # 通过 imshow() 函数在 Matplotlib 查看器上直接根据像素点绘制出相应的图像
    plt.imshow(frame, cmap=color_map)
    plt.title(predicted_label)
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()