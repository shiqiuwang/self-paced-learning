# encoding:utf-8
# author:WangQiuShi
# data:2020/9/21 9:56
import numpy as np


def spl(sample_loss_dict, lam=0.15):
    selected_sample = []
    sorted_sample = sorted(sample_loss_dict.items(), key=lambda s: s[1], reverse=False)
    for item in sorted_sample:
        if item[1] < lam:
            selected_sample.append(item[0])
        else:
            break
    return selected_sample


# def spld(loss, group_member_ship, lam=0.01, gamma=0.3):
#     group_idx = set(group_member_ship)
#     b = len(group_idx)
#     selected_idx = [0] * len(loss)
#     selected_scores = [0] * len(loss)
#     for j in range(b):
#         idx_ingroup = np.where(group_member_ship == group_idx[j]).tolist()
#         loss_ingroup = loss[idx_ingroup]
#         randk_ingroup=


if __name__ == '__main__':
    loss = [0.05, 0.12, 0.12, 0.12, 0.15, 0.40, 0.17, 0.18, 0.35, 0.15, 0.16, 0.20, 0.50, 0.28]
    samples_loss_dict = dict()
    for i in range(97, 111):
        samples_loss_dict[chr(i)] = loss[i - 97]
    print("When lambda=0.15, SPL selects:{}".format(" ".join(spl(samples_loss_dict, lam=0.15))))
