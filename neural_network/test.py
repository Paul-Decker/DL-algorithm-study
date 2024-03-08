import numpy as np
# 创建2×3多维数组
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a, end="\n\n")
'''
[[1 2]
 [3 4]
 [5 6]]
'''
# 转化为3×2多维数组
b = a.flatten()
print(b)
print(b.ndim)
print(b.shape)
'''
[[1 2 3]
 [4 5 6]]
'''