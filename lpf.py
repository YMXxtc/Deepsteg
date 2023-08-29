import torch
import numpy as np
import torch.nn as nn
import cv2


lpf3 = np.array([
        [1, 1, 1],
        [1, 8, 1],
        [1, 1, 1]
    ], dtype=np.float32) / 16

lpf3_v2 = np.array([
        [1, 2, 1],
        [2, 4, 3],
        [1, 2, 1]
    ], dtype=np.float32) / 16

lpf5 = np.array([
        [1, 1, 1, 1, 1],
        [1, 4, 4, 4, 1],
        [1, 4, 48, 4, 1],
        [1, 4, 4, 4, 1],
        [1, 1, 1, 1, 1]
    ], dtype=np.float32) / 64


class LPF_mean(nn.Module):
    def __init__(self, size=3):
        super(LPF_mean, self).__init__()

        self.size = size

        mean_filter = np.array(np.ones((self.size,self.size)) / (self.size*self.size), dtype=np.float32)
        # print(mean_filter)
        filter_list = [mean_filter]
        filter_list = np.array(filter_list)
        # filter_list = [mean_filter]

        lpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, self.size, self.size), requires_grad=False)

        self.lpf = nn.Conv2d(1, len(filter_list), kernel_size=self.size, padding=self.size//2, bias=False, padding_mode='replicate')
        self.lpf.weight = lpf_weight

    def forward(self, input):
        output = self.lpf(input)

        return output

class LPF3(nn.Module):
    def __init__(self):
        super(LPF3, self).__init__()

        filter_list = [lpf3]
        filter_list = np.array(filter_list)

        lpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 3, 3), requires_grad=False)

        self.lpf = nn.Conv2d(1, len(filter_list), kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.lpf.weight = lpf_weight

    def forward(self, input):
        output = self.lpf(input)

        return output

class LPF5(nn.Module):
    def __init__(self):
        super(LPF5, self).__init__()

        filter_list = [lpf5]
        filter_list = np.array(filter_list)

        lpf_weight = nn.Parameter(torch.Tensor(filter_list).view(len(filter_list), 1, 5, 5), requires_grad=False)

        self.lpf = nn.Conv2d(1, len(filter_list), kernel_size=5, padding=1, bias=False, padding_mode='replicate')
        self.lpf.weight = lpf_weight

    def forward(self, input):
        output = self.lpf(input)

        return output