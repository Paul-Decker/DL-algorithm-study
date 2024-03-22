import numpy as np
'''
points是个二维数据结构
points[i, 0]表示第i个点的x值
points[i, 1]表示第i个点的y值
'''
# y = wx + b
# 计算损失值 loss = (y-(w*x+b))^2
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

'''
loss = (y-(w*x+b))^2 = (w*x+b-y)^2
b_gradient = 2(w*x+b-y)*1
w_gradient = 2(w*x+b-y)*x
'''
# 对参数进行一次更新
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, N):
        x = points(i, 0)
        y = points(i, 1)
        # 计算梯度值
        # 计算所有点的梯度值，求平均
        b_gradient += (2/N) * (((w_current*x)+b_current) - y)
        w_gradient += (2/N) * (((w_current*x)+b_current) - y) * x
    # 更新参数
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

# 梯度下降
def gradient_descent_runner(points, starting_b, starting_w, learningRate, num_iteration):
    b = starting_b
    w = starting_w
    for _ in range(num_iteration):
        b, w = step_gradient(b, w, np.array(points), learningRate)
    # 此时返回的是最优的参数b和w
    return [b, w]

def run():
    # 读取数据
    points = np.genfromtxt('data.csv', delimiter=',')
    # 设置学习率
    learningRate = 0.001
    # 初始化参数
    initial_b = 0
    initial_w = 0
    # 设置迭代次数
    num_iteration = 1000
    print(f'starting gradient descent at b = {initial_b}, w = {initial_w}, '
        + f'error/loss = {compute_error_for_line_given_points(initial_b, initial_w, points)}')
    print('Running...')
    [b, m] = gradient_descent_runner(points, initial_b, initial_w, learningRate, num_iteration)
    print(f'After {num_iteration} iterations b = {b}, w = {w}, '
        + f'error = {compute_error_for_line_given_points(b, m, points)}')
    
    if __name__ == 'main':
        run()