# encoding:utf-8
# author:WangQiuShi
# data:2020/9/22 9:18

import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

torch.manual_seed(2020)
np.random.seed(2020)

# 设定超参数
EPOCH = 10
lam = 0.03
gamma = 0.2
lr = 0.001

# 自步学习参数
u1 = 1.01
u2 = 1.01


# spld
def spld(loss, group_member_ship, lam, gamma):
    groups_labels = np.array(list(set(group_member_ship)))
    b = len(groups_labels)
    selected_idx = []
    selected_score = [0] * len(loss)
    for j in range(b):
        idx_in_group = np.where(group_member_ship == groups_labels[j])[0]
        # print(idx_in_group)
        loss_in_group = []
        # print(type(idx_in_group))
        for idx in idx_in_group:
            loss_in_group.append(loss[idx])
        idx_loss_dict = dict()
        for i in idx_in_group:
            idx_loss_dict[i] = loss[int(i)]
        sorted_idx_in_group = sorted(idx_loss_dict.keys(), key=lambda s: idx_loss_dict[s])
        sorted_idx_in_group_arr = np.array(sorted_idx_in_group)

        # print(sorted_idx_in_group_arr)

        for (i, ii) in enumerate(sorted_idx_in_group_arr):
            if loss[ii] < (lam + gamma / (np.sqrt(i + 1) + np.sqrt(i))):
                selected_idx.append(ii)
            else:
                pass
            selected_score[ii] = loss[ii] - (lam + gamma / (np.sqrt(i + 1) + np.sqrt(i)))

    selected_idx_arr = np.array(selected_idx)
    selected_idx_and_new_loss_dict = dict()
    for idx in selected_idx_arr:
        selected_idx_and_new_loss_dict[idx] = selected_score[idx]

    sorted_idx_in_selected_samples = sorted(selected_idx_and_new_loss_dict.keys(),
                                            key=lambda s: selected_idx_and_new_loss_dict[s])

    sorted_idx_in_selected_samples_arr = np.array(sorted_idx_in_selected_samples)
    return sorted_idx_in_selected_samples_arr


# 数据集
x = torch.rand((20, 1), dtype=torch.float32) * 10  # (20,1)print(x)
# print(x.size())
y = (2 * x + 5) + torch.rand((20, 1), dtype=torch.float32)  # （20,1）
# print(y)

# 初始化线性回归参数
w = torch.randn([1], requires_grad=True)  # w一般初始化为正态随机分布里面的一个值
b = torch.zeros([1], requires_grad=True)  # b一般初始化为0

# 聚类
model = KMeans(n_clusters=4, max_iter=10)
model.fit(x)

samples_label = model.labels_  # 每一个样本对应的类别 ndarray
samples_label_arr = np.array(list(set(samples_label)))
# print(samples_label_arr)
# print(samples_label)
# print(type(samples_label))

# 初始化v_star
v_star = torch.randint(low=0, high=2, size=(len(x.numpy()),), dtype=torch.float32)
# print(v_star.sum())
# print(v_star)

for epoch in range(1000):
    wx = torch.mul(w, x)
    # print(wx)
    y_pred = torch.add(wx, b)
    # print(y_pred)

    # 计算MSE loss
    loss = (0.5 * (y - y_pred) ** 2).reshape(20, )
    # print(loss.mean())

    # print(loss)
    # print(type(loss))
    # print(torch.matmul(v_star,loss))
    # loss1 = torch.matmul(v_star, loss) - lam * v_star.sum()
    # # print(loss1)
    # loss2 = torch.tensor(0, dtype=torch.float32)
    # # print(loss2)
    # # print(type(loss2))
    # for group_id in samples_label_arr:
    #     idx_for_each_group = np.where(samples_label == group_id)[0]
    #     loss_for_each_group = torch.tensor(0, dtype=torch.float32)
    #     for idx in idx_for_each_group:
    #         loss_for_each_group += (v_star[idx] ** 2)
    #     loss2 += torch.sqrt(loss_for_each_group)
    # # 计算E
    # E = loss1 - gamma * loss2
    #
    # # print(E)

    # 反向传播
    loss.backward()
    print(loss.mean().grad)

    # print(w.grad)
    # print(b.grad)

    # 更新w,b,即模型的参数
    w.data.sub_(lr * w.grad)
    b.data.sub_(lr * b.grad)
    # print(w)
    # print(b)

    # 梯度清零
    w.grad.zero_()
    b.grad.zero_()

    new_y_pred = w * x + b
    new_loss = (0.5 * (y - new_y_pred) ** 2)
    selected_idx_arr = spld(new_loss.reshape(new_loss.size()[0], ), samples_label, lam, gamma)
    # print(selected_idx_arr)

    v_star = torch.zeros((new_loss.size()[0],), dtype=torch.float32)

    for selected_idx in selected_idx_arr:
        v_star[selected_idx] = 1
    # print(v_star)
    # print("-------------------")
    lam = u1 * lam
    gamma = u2 * gamma
