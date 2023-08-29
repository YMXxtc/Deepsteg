
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

from ResUNet1NoSkip import UNet
from hpf import *
from lpf import *

import torchvision
from thop import profile


TS_PARA = 60
PAYLOAD = 0.4

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


def myParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='bows|bossbase|szu',default='szu')
    parser.add_argument('--dataroot', help='path to dataset',default='/data/ymx/ymx/SZU-cover-resample-256/')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')

    parser.add_argument('--niter', type=int, default=72, help='number of epochs to train for')
    parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0002')

    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--outf', default='Res', help='folder to output images and model checkpoints')
    
    parser.add_argument('--config', default='deep_steg', help='config for training')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('-g','--gpuid', default='0,1', type=str, help='gpuid to use')

    parser.add_argument('--alpha1', default=10, type=float, help='alpha1 for g_l1')
    parser.add_argument('--alpha2', default=1.0, type=float, help='alpha2 for g_l2')
    parser.add_argument('--alpha3', default=1e-4, type=float, help='alpha3 for g_l3')  #1e-4
    parser.add_argument('--alpha4', default=1000, type=float, help='alpha4 for g_l4')  

    parser.add_argument('--flat_rate', default=0.5, type=float, help='choose the rate for flat region')

    opt = parser.parse_args()
    return opt


class SZUDataset256(data.Dataset):
    def __init__(self, root, transforms = None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]

        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]

        label = np.array([0, 1], dtype='int32')
        
        data = cv2.imread(img_path,-1)
        rand_ud = np.random.rand(256,256)
        sample = {'data':data, 'rand_ud':rand_ud, 'label':label, 'index':index}
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample

    def __len__(self):
        return len(self.imgs)


class ToTensor():
  def __call__(self, sample):
    data,rand_ud, label, index = sample['data'], sample['rand_ud'], sample['label'], sample['index']

    data = np.expand_dims(data, axis=0)
    data = data.astype(np.float32)
    rand_ud = rand_ud.astype(np.float32)
    rand_ud = np.expand_dims(rand_ud,axis = 0)
    
    data = data / 255.0

    new_sample = {
      'data': torch.from_numpy(data),
      'rand_ud': torch.from_numpy(rand_ud),
      'label': torch.from_numpy(label).long(),
      'index': index,
    }

    return new_sample


# custom weights initialization called on netG and netD
def weights_init_g(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d) and m.weight.requires_grad:
            m.weight.data.normal_(0., 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0., 0.02)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0., 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

def setLogger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

## residual guidance
class residual_guidance(nn.Module):
    def __init__(self):
       super(residual_guidance,self).__init__()

    def forward(self, cover, m):
        k1 = HPF_kb3().cuda()
        k2 = HPF_kv5().cuda()

        LPF_hf = LPF_mean(3).cuda()
        LPF_m1 = LPF_mean(11).cuda()
        LPF_m2 = LPF_mean(7).cuda()

        hf_x1 = torch.abs(k1(cover))          
        hf_x1 = torch.sum(hf_x1, dim=1, keepdim=True)
        hf_max = torch.max(hf_x1); hf_min = torch.min(hf_x1)    #resdidual normalization
        normal_hf_x1 = (hf_x1 - hf_min) / (hf_max - hf_min) / 8 
        normal_hf_x1 = LPF_hf(normal_hf_x1)

        hf_x2 = torch.abs(k2(cover))          
        hf_x2 = torch.sum(hf_x2, dim=1, keepdim=True)
        hf_max2 = torch.max(hf_x2); hf_min2 = torch.min(hf_x2)    
        normal_hf_x2 = (hf_x2 - hf_min2) / (hf_max2 - hf_min2) / 8 
        normal_hf_x2 = LPF_hf(normal_hf_x2)

        epsilon = 1e-5   

        lf_m1 = LPF_m1(torch.abs(m))
        lf_m2 = LPF_m2(torch.abs(m))
        lf_m = (lf_m1+lf_m2) / 2

        err = (1 / (normal_hf_x1 + epsilon) + 1 / (normal_hf_x2 + epsilon)) * lf_m 

        err = torch.mean(err)
        return err

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['G_loss']))

    y1 = hist['G_loss']
    y2 = hist['g_l1']
    y3 = hist['g_l2']
    y4 = hist['g_l3']
    y5 = hist['g_l4']
    
    plt.plot(x, y1, label='G_loss')
    plt.plot(x, y2, label='g_l1')
    plt.plot(x, y3, label='g_l2')
    plt.plot(x, y4, label='g_l3')
    plt.plot(x, y5, label='g_l4')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path1 = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path1)

    plt.close()


# calculate 3*3 local variance
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


def main():
    
    opt = myParseArgs()
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuid

    config = opt.config

    cudnn.benchmark = True
    LOG_PATH = os.path.join(opt.outf, 'model_log_'+ config) #opt.config
    setLogger(LOG_PATH, mode = 'w')

    
    transform = transforms.Compose([ToTensor(),])
    dataset = SZUDataset256(opt.dataroot, transforms=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers),drop_last = True)

    
    netG = UNet()
    netG = nn.DataParallel(netG)
    netG = netG.cuda()
    #print(netG)
    netG.apply(weights_init_g)

    start_epoch = 0
    if opt.netG != '':
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(opt.netG))
        logging.info('-' * 8)
        netG.load_state_dict(torch.load(opt.netG))
    # print(netG)

    criterion = residual_guidance().cuda()

    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))

    scheduler_G = StepLR(optimizerG, step_size=20, gamma=0.4)  

    train_hist = {}
    train_hist = {}
    train_hist['g_l1'] = []
    train_hist['g_l2'] = []
    train_hist['g_l3'] = []
    train_hist['g_l4'] = []
    train_hist['G_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []

    
    start_time = time.time()
    

    for epoch in range(start_epoch, opt.niter):

        netG.train()

        scheduler_G.step()


        epoch_start_time = time.time()
        for i,sample in enumerate(dataloader,0):
            
            optimizerG.zero_grad()

            data, rand_ud, label, index = sample['data'], sample['rand_ud'],sample['label'],sample['index']
            cover, n, label = data.cuda(), rand_ud.cuda(), label.cuda()

            p = netG(cover)   

            p_plus = p/2.0 + 1e-5
            p_minus = p/2.0 + 1e-5
            
            m =  - 0.5 * torch.tanh(TS_PARA * (p - 2 * n)) + 0.5 * torch.tanh(TS_PARA * (p - 2 * (1 - n))) 
            
            stego = cover*255.0 + m

            C = -(p_plus * torch.log2(p_plus) + p_minus*torch.log2(p_minus)+ (1 - p+1e-5) * torch.log2(1 - p+1e-5))
            
            label = label.reshape(-1)

            ## residual distance guidance loss
            hpf = HPF_srm30().cuda()

            hpf_cover = hpf(cover)
            hpf_stego = hpf(stego / 255.0) # 

            hpf_diff = torch.abs(hpf_stego - hpf_cover)
            hpf_diff = torch.mean(hpf_diff, dim=1, keepdim=True)

            g_l1 = -torch.mean(p * (-hpf_diff))
            g_l1 = opt.alpha1 * g_l1


            ## residual guidance loss
            g_l2 = criterion(data.cuda(), m.cuda())
            g_l2 = opt.alpha2 * g_l2


            ## complex region guidance loss
            hpf_mask_cover = get_local_variance(cover)
            rate = 0.5
 
            quant = torch.quantile(hpf_mask_cover, rate, dim=2, keepdim=True)
            quant = torch.quantile(quant, rate, dim=3, keepdim=True)
            mask = (hpf_mask_cover < quant).type(torch.float32)   #1 stands for flat region

            g_l4 = torch.mean(mask * p)   
            g_l4 = opt.alpha4 * g_l4
            
            ## embedding loss
            g_l3 = torch.mean((C.sum(dim = (1,2,3)) - 256 * 256 * PAYLOAD) ** 2)
            g_l3 = opt.alpha3 * g_l3


            errG =  g_l1 + g_l2 + g_l3 + g_l4 
            changeRate = torch.mean(torch.abs(m))


            if epoch > 0:
                train_hist['G_loss'].append(errG.item())
                train_hist['g_l2'].append(g_l2.item())
                train_hist['g_l1'].append(g_l1.item())
                train_hist['g_l3'].append(g_l3.item())
                train_hist['g_l4'].append(g_l4.item())
            
            errG.backward()
            optimizerG.step()
            
            logging.info('Epoch: [%d/%d][%d/%d] g_l1: %.4f g_l2: %.4f g_l3: %.4f g_l4: %.4f Loss_G: %.4f C:%.4f changeRate: %.4f ' % (epoch, opt.niter-1, i, len(dataloader), g_l1.item(), g_l2.item(), g_l3.item(), g_l4.item(), errG.item(), C.sum().item()/opt.batchSize, changeRate.item()))


        train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            
        # do checkpointing
        if (epoch+1)%10 == 2 and (epoch + 1) >= (opt.niter - 14):
        
            torch.save(netG.state_dict(), '%s/netG_epoch_%s_%d.pth' % (opt.outf, config, epoch+1))  #opt.config
        
        loss_plot(train_hist, opt.outf, model_name = opt.outf + config)  #opt.config

    train_hist['total_time'].append(time.time() - start_time)
    logging.info("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(train_hist['per_epoch_time']),
                                                                            epoch, train_hist['total_time'][0]))
    logging.info("Training finish!... save training results")

if __name__ == '__main__':
    main()
