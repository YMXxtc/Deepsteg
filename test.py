import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import scipy.io as sio
import logging
from pathlib import Path
from torch.optim.lr_scheduler import StepLR

from hpf import *

def get_local_variance(cover_batch):
    vari_batch = torch.zeros(cover_batch.shape).cuda()

    for i in range(cover_batch.shape[0]): 
        cover = cover_batch[i]

        cover = torch.unsqueeze(cover, dim=0)

        pad = nn.ReplicationPad2d(padding=(1,1,1,1)).cuda()
        cover_pad = pad(cover)
        # print(cover_pad.shape)

        #create patch
        unfold = nn.Unfold(kernel_size=3, stride=1).cuda()

        cover_blocks = unfold(cover_pad)
        # print(cover_blocks.shape)

        cover_blocks = cover_blocks.reshape(3*3, -1)

        vari_blocks = torch.var(cover_blocks, dim=0)
        # print(vari_blocks.shape)

        vari_batch[i] = vari_blocks.reshape(cover.shape)
    
    # print(vari_batch.shape)
    return vari_batch


img_path = '/data/ymx/ymx/BOSSBase/BossBase-1.01-cover-resample-256/{}.pgm'
save_path = '/data/ymx/ymx/mine/Res/a/{}.png'

flat_rate = 0.5
for i in range(18,19):
    cover = cv2.imread(img_path.format(i+1),-1)
    # cover = cv2.imread('/data/ymx/ymx/mine/Res/a/cover.pgm',-1)
    print(cover.shape)
    data = np.expand_dims(cover, axis=0)
    data = data.astype(np.float32)

    cover = torch.from_numpy(data).cuda()
    cover = cover.reshape(1,1,256,256)

    # hpf_mask = HPF_kb3().cuda()
    # hpf_mask_cover = torch.abs(hpf_mask(cover))

    #get local variance 
    hpf_mask_cover = get_local_variance(cover)

    S = hpf_mask_cover / torch.max(hpf_mask_cover) * 255.0
    S = S.reshape(256,256)
    cv2.imwrite('/data/ymx/ymx/mine/Res/a/S.png', np.array(S.cpu()))
 
    quant = torch.quantile(hpf_mask_cover, flat_rate, dim=2, keepdim=True)
    quant = torch.quantile(quant, flat_rate, dim=3, keepdim=True)
    # print(quant)
    # print(hpf_mask_cover)
    mask = (hpf_mask_cover < quant).type(torch.uint8)   #1 stands for flat region
    
    mask = mask.reshape(256,256)
    # print(mask.shape)

    # cv2.imwrite(save_path.format(i+1), np.array(mask.cpu()*255.0))
    cv2.imwrite('/data/ymx/ymx/mine/Res/a/F.png', np.array(mask.cpu()*255.0))