import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TowDim_section = slice(0,2)
ThreeDim_section = slice(0,3)

digits = load_digits()
# 加载数据集
# print(digits.keys())
# 查看数据的键
# print(digits.data.shape)
# (1797, 64)  说明有1797个样本，64个特征
# print(digits.target_names)
# 标签 对应0到9
# print(digits.images.shape)
# (1797, 8, 8)  1797张8*8的图片，对应64个特征

basic_data = np.array(digits.data)
# print(basic_data .shape)

mean_data = np.mean(basic_data,axis=0)
# 对原始数据求每一维度的均值，所以axis=0
# print(mean_data.shape)
data = basic_data-mean_data
# 去中心化

cov_matrix =np.cov(data,rowvar=False)
# 计算协方差,列是维度,所以rowvar为False
# 也可以是cov_matrix = 1/(1797-1)*np.mat(data.T)*np.mat(data)
# print(cov_matrix)

eigen_value,eigen_vector = np.linalg.eig(np.mat(cov_matrix))
# 求协方差矩阵的特征值与特征向量
# print(eigen_value.shape,eigen_vector.shape)
# (64,) (64, 64),说明每特征向量是列向量
# print(np.mat(eigen_value).shape,type(eigen_vector))

index = np.argsort(-eigen_value)
# 对特征值进行降序排序

TowDim_index = index[TowDim_section]
ThreeDim_index = index[ThreeDim_section]

# print(TowDim_index)
# 选取特征值

TowDim_value = eigen_value[TowDim_index]
ThreeDim_value = eigen_value[ThreeDim_index]
# 选取对应的特征值

TowDim_vector = eigen_vector[:,TowDim_index]
ThreeDim_vector = eigen_vector[:,ThreeDim_index]
# 选取对应的特征向量
# print(type(TowDim_vector))
# print(TowDim_vector)
# print(TowDim_vector.shape,ThreeDim_vector.shape)

TowDim_data = np.dot(data,TowDim_vector)
ThreeDim_data = np.dot(data,ThreeDim_vector)
# 求得对应维度的数据
