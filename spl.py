# encoding:utf-8
# author:WangQiuShi
# data:2020/9/21 16:45
import numpy as np


def spl(loss, lam):
    selected_idx = []
    for idx, val in enumerate(loss):
        if val < lam:
            selected_idx.append(idx)
    selected_idx_arr = np.array(selected_idx)
    return selected_idx_arr


if __name__ == '__main__':
    samples = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
    loss = [0.05, 0.12, 0.12, 0.12, 0.15, 0.40, 0.17, 0.18, 0.35, 0.15, 0.16, 0.20, 0.50, 0.28]
    selected_samples = []

    selected_idx_arr = spl(loss,0.15)

    print(selected_idx_arr)
    for selected_idx in selected_idx_arr:
        selected_samples.append(samples[selected_idx])
    print("selected samples are:{}".format(",".join(selected_samples)))
