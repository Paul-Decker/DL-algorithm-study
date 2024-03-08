import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient


class MultilayerPerception:
    def __init__(self, data, labels, layers, normalize_data=False):
        # 数据预处理
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]
        self.data = data_processed
        # 标签（即训练数据的“正确答案”）
        self.labels = labels
        # layers表示各层的神经元：layers[0]:输入层的784个神经元，layers[1]:隐层1的25个神经元，layers[3]:输出层的10个分类结果
        self.layers = layers    # 输入层：28×28×1=784个像素点（神经元） 隐层1：25个神经元（将784个特征转化为25个） 输出层：10个分类任务
        # 是否对数据进行归一化操作
        self.normalize_data = normalize_data
        # 参数Θ的初始化操作
        self.thetas = MultilayerPerception.thetas_init(layers)

    # 训练模块
    def train(self, max_iterations=1000, alpha=0.1):
        unrolled_theta = MultilayerPerception.thetas_unroll(self.thetas)
        # 梯度下降
        (optimized_theta, cost_history) = MultilayerPerception.gradient_descent(self.data, self.labels, unrolled_theta, self.layers, max_iterations, alpha)
        
        self.thetas = MultilayerPerception.thetas_roll(optimized_theta, self.layers)

        return self.thetas, cost_history

    @staticmethod
    # 参数初始化
    def thetas_init(layers):
        # 层数（3层）
        num_layers = len(layers)
        # 构建参数(包括了权重参数和偏置参数)
        thetas = {}
        for layer_index in range(num_layers - 1):
            """
            会执行两次，得到两组参数矩阵: 25*785, 10*26
            """
            # 输入层
            in_count = layers[layer_index]
            # 输出层
            out_count = layers[layer_index+1]
            # 这里需要考虑上偏置项，所以需要+1，偏置的个数与输出结果是一致的
            thetas[layer_index] = np.random.rand(out_count, in_count+1)*0.05  # 随机进行参数（权重参数和偏置参数）初始化操作，值尽量小一点
        return thetas

    @staticmethod
    # 梯度下降（前向传播、反向传播）
    def gradient_descent(data, labels, unrolled_theta, layers, max_iterations, alpha):
        # 1. 计算损失值
        # 2. 计算梯度值
        # 3. 更新梯度值

        optimized_theta = unrolled_theta
        cost_history = []

        for _ in range(max_iterations):
            # 计算损失值
            cost = MultilayerPerception.cost_function(data, labels, MultilayerPerception.thetas_roll(optimized_theta, layers), layers)
            cost_history.append(cost)
            # 计算梯度值
            theta_gradient = MultilayerPerception.gradient_step(data, labels, optimized_theta, layers)
            # 更新梯度值
            optimized_theta = optimized_theta - alpha * theta_gradient
        return optimized_theta, cost_history

    @staticmethod
    def gradient_step(data, labels, optimized_theta, layers):
        theta = MultilayerPerception.thetas_roll(optimized_theta, layers)
        thetas_rolled_gradients = MultilayerPerception.back_propagation(data, labels, theta, layers)
        thetas_unrolled_gradients = MultilayerPerception.thetas_unroll(thetas_rolled_gradients)
        return thetas_unrolled_gradients

    @staticmethod
    # 反向传播
    def back_propagation(data, labels, thetas, layers):
        num_layers = len(layers)
        (num_examples, num_features) = data.shape
        # 输出层的10个分类
        num_label_types = layers[-1]
        # δ计算每层对结果的误差
        deltas = {}
        # δ的初始化操作
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            deltas[layer_index] = np.zeros((out_count, in_count+1))  # 25*725   10*26

        for example_index in range(num_examples):
            # 当前层的输入
            layers_inputs = {}
            # 当前层经过激活函数后的结果
            layers_activations = {}
            # 第0层
            layers_activation = data[example_index, :].reshape((num_features, 1))  # 785*1
            layers_activations[0] = layers_activation
            # 逐层计算（从第一层开始）
            for layer_index in range(num_layers - 1):
                layer_theta = thetas[layer_index]  # 得到当前权重参数值：25*785  10*26
                layer_input = np.dot(layer_theta, layers_activation)  # 第一次得到25*1（中间隐层），第二次得到10*1（结果输出层）
                # 加上偏置参数
                layers_activation = np.vstack((np.array([[1]]), sigmoid(layer_input)))
                layers_inputs[layer_index + 1] = layer_input  # 后一层计算结果
                layers_activations[layer_index + 1] = layers_activation  # 后一层经过激活函数后的结果
            # 去掉偏置参数
            output_layer_activation = layers_activation[1:, :]

            delta = {}
            # 标签处理
            bitwise_label = np.zeros((num_label_types, 1))
            bitwise_label[labels[example_index][0]] = 1
            # 计算输出层与真实值之间的差异
            delta[num_layers - 1] = output_layer_activation - bitwise_label

            # 遍历循环L L-1 L-2 ...
            for layer_index in range(num_layers - 2, 0, -1):
                layer_theta = thetas[layer_index]
                next_delta = delta[layer_index + 1]
                layer_input = layers_inputs[layer_index]
                # 加上偏置项
                layer_input = np.vstack((np.array((1)), layer_input))

                # 按照公式计算
                delta[layer_index] = np.dot(layer_theta.T, next_delta)*sigmoid_gradient(layer_input)
                # 过滤掉偏置参数
                delta[layer_index] = delta[layer_index][1:, :]
            
            for layer_index in range(num_layers - 1):
                layer_delta = np.dot(delta[layer_index+1], layers_activations[layer_index].T)
                deltas[layer_index] = deltas[layer_index] + layer_delta  # 第一次25*785  第二次10*26

        for layer_index in range(num_layers - 1):
            deltas[layer_index] = deltas[layer_index] * (1/num_examples)
        
        return deltas

    @staticmethod
    # 损失函数
    def cost_function(data, labels, thetas, layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        num_labels = layers[-1]

        # 前向传播走一次
        predictions = MultilayerPerception.feedforward_propagation(data, thetas, layers)
        # 制作标签，每一个样本的标签都得是one-hot 
        bitwise_labels = np.zeros((num_examples, num_labels))
        for example_index in range(num_examples):
            bitwise_labels[example_index][labels[example_index][0]] = 1
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        bit_not_set_cost = np.sum(np.log(1 - predictions[bitwise_labels == 0]))
        cost = (-1 / num_examples) * (bit_set_cost + bit_not_set_cost)
        return cost

    @staticmethod
    # 前向传播
    def feedforward_propagation(data, thetas, layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        in_layer_activation = data

        # 逐层计算
        for layer_index in range(num_layers - 1):
            # 获取当前层的参数
            theta = thetas[layer_index]
            # 输入层有784个特征值，数据预处理后会变为785个
            # 矩阵in_layer_activation的形状：n×785（n为训练数据的个数，即有n张图片要传进来）
            # theta.T表示将theta矩阵转置，转置后的形状为：785×25
            out_layer_activation = sigmoid(np.dot(in_layer_activation, theta.T))
            # 正常计算完后是num_examples×25，但是要考虑偏置项，编程num_examples×26
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))
            # 这一层的输出作为下一层的输入
            in_layer_activation = out_layer_activation
        
        # 返回输出层的结果, 结果中不要偏置项了
        return in_layer_activation[:, 1:]

    @staticmethod
    # 将2维矩阵展开成1维数组，方便后续更新数值
    def thetas_unroll(thetas):
        # θ的个数（2个）
        num_theta_layers = len(thetas)
        unrolled_theta = np.array([])
        for theta_layer_index in range(num_theta_layers):
            # hstack函数：对矩阵进行拼接；flatten函数：展开矩阵
            unrolled_theta = np.hstack((unrolled_theta, thetas[theta_layer_index].flatten()))
        return unrolled_theta

    @staticmethod
    # 将展开的矩阵，还原为原来的形状
    def thetas_roll(unrolled_thetas, layers):
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]

            thetas_width = in_count + 1
            thetas_height = out_count
            thetas_volume = thetas_width * thetas_height
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layer_theta_unrolled.reshape((thetas_height, thetas_width))

            unrolled_shift = unrolled_shift + thetas_volume

        return thetas