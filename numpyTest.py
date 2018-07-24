# coding=utf-8
"""
created on:2018/4/22
author:DilicelSten
target:Learn numpy
"""
import numpy as np


# -------------------------------------ndarray------------------------------------------

lst = [[1, 2, 3], [4, 5, 6]]
print type(lst)

# 列表转ndarray
np_lst = np.array(lst)
print type(np_lst)

# 定义类型
np_lst = np.array(lst, dtype=np.float)

print np_lst.shape
print np_lst.ndim  # 维数
print np_lst.dtype  # 类型
print np_lst.itemsize  # 每个的大小
print np_lst.size  # 大小

# ---------------------------------------some kinds of array------------------------------------------
# 数值的初始化
print np.zeros([2, 4])  # 全0
print np.ones([3, 5])   # 全1

# 生成随机数
print np.random.rand(2, 4)
print np.random.rand()
# 随机整数
print np.random.randint(1, 10, 3)
# 标准正态分布
print np.random.randn(2, 4)
# 生成指定值
print np.random.choice([1, 2, 3, 4, 6, 7, 8, 9])

# 生成贝塔分布1-10 100个
print np.random.beta(1, 10, 100)  # 使用random生成各种分布

# -------------------------------------operations------------------------------------------

# 生成等差数列
lst = np.arange(1, 11).reshape([2, 5])  # 2行5列，5可以缺省成-1
# 自然指数
print np.exp(lst)
# 指数的平方
print np.exp2(lst)
# 开方
print np.sqrt(lst)
# 三角函数
print np.sin(lst)
# 对数
print np.log(lst)
# 求和
lst = np.array([[1, 2, 3], [4, 5, 6]])
print lst.sum(axis=0)  # axis指定维度
# 最大最小
print lst.max(axis=1)
print lst.min(axis=0)
# 拆解/拉直——>多维数组变一维数组
print lst.ravel()  # 返回的只是数组的视图
print lst.flatten()  # 返回的是真实的数组

# 两个的操作
lst1 = np.array([10, 20, 30, 40])
lst2 = np.array([1, 2, 3, 4])
# +-*/
print lst1+lst2
# 平方
print lst1**2
# 点积
print np.dot(lst1.reshape([2, 2]), lst2.reshape([2, 2]))
# 合成
print np.concatenate((lst1, lst2), axis=1)  # 0水平叠加 1垂直叠加

# 堆叠
print np.vstack((lst1, lst2))  # 垂直叠加
print np.hstack((lst1, lst2))  # 水平叠加

# 分开
print np.split(lst1, 2)  # 分成两份
# 拷贝
print np.copy(lst1)

# ------------------------------------linear algebra-------------------------------------------
# 引入模块
from numpy.linalg import *

# 生成单位矩阵
print np.eye(3)
lst = np.array([[1, 2], [3, 4]])
# 求矩阵的逆
print inv(lst)
# 转置矩阵T
print lst.transpose()
# 行列式
print det(lst)
# 特征值和特征向量
print eig(lst)
# 解多元一次方程
a = np.array([[1, 2, 1], [2, -1, 3], [3, 1, 2]])
b = np.array([7, 7, 18])
x = solve(a, b)
print x

# ------------------------------------other-------------------------------------------
# 求相关系数
print np.corrcoef([1, 0, 1], [0, 2, 1])
# 生成函数
print np.poly1d([2, 1, 3])
