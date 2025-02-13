import os
import sys
<<<<<<< HEAD
import tqdm
=======
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
sys.path.append('.')
import torch
import torch.nn.functional as F
import numpy as np
from utils import read
from metric import calculate_psnr, calculate_ssim
<<<<<<< HEAD
from models.My_S import Model
from collections import Counter
from torch.utils.data import DataLoader
from datasets import XVFI_Test_Dataset, Vimeo90K_Test_Dataset, UCF_Test_Dataset, SNU_Test_Dataset

=======
from models.IFRNet import Model
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
# from models.IFRNet_L import Model
# from models.IFRNet_S import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

<<<<<<< HEAD
model = Model()
model.load_state_dict(torch.load('../weights/newS.pth'))
=======

model = Model()
model.load_state_dict(torch.load('checkpoints/IFRNet/IFRNet_Vimeo90K.pth'))
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
# model.load_state_dict(torch.load('checkpoints/IFRNet_large/IFRNet_L_Vimeo90K.pth'))
# model.load_state_dict(torch.load('checkpoints/IFRNet_small/IFRNet_S_Vimeo90K.pth'))
model.eval()
model.cuda()

<<<<<<< HEAD
divisor = 64 #20
scale_factor = 1.0 #0.8


class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """

=======
divisor = 20
scale_factor = 0.8

class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
    def __init__(self, dims, divisor=divisor):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
<<<<<<< HEAD
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
=======
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

<<<<<<< HEAD
    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


# Replace the 'path' with your SNU-FILM dataset absolute path.
path = '/home/luolab/xzh/IFRNet-main/data/SNU-FILM/'
prefix_path = '/home/luolab/xzh/IFRNet-main'

test_files = ["test-medium.txt","test-hard.txt","test-extreme.txt"]
for j in range(4):
    test_file = test_files[j] # test-easy.txt, test-medium.txt, test-hard.txt, test-extreme.txt
    mask_ratios = [[], [], []]
    psnr_list = []
    ssim_list = []
    file_list = []
    MACs_list = []
    with open(os.path.join(path, test_file), "r") as f:
        for line in f:
            line = line.strip()
            file_list.append(line.split(' '))

    F_sum = 0
    sizes = []
    for line in file_list:
        #print(os.path.join(prefix_path, line[0]))
        I0_path = os.path.join(prefix_path, line[0])
        I1_path = os.path.join(prefix_path, line[1])
        I2_path = os.path.join(prefix_path, line[2])
        I0 = read(I0_path)
        I1 = read(I1_path)
        I2 = read(I2_path)
        I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
        I1 = (torch.tensor(I1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
        I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
        #padder = InputPadder(I0.shape)
        #I0, I2 = padder.pad(I0, I2)
        I = torch.cat([I0,I1,I2],dim=1)
        I = F.interpolate(I,size=(768,1280),mode='bilinear')
        I0, I1, I2 = I[:,:3], I[:,3:6], I[:,6:9]
        embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

        sizes.append(I0.shape)

        #imgt_pred,mask_ratio,_,_,_ = model.test_mask(I0, I1, I2, embt, scale_factor=scale_factor)
        with torch.no_grad():
            FModel, imgt_pred, mask_ratio, _, _, _ = model.get_dynamic_MACs(I0, I1, I2, embt, scale_factor=scale_factor)
        F_sum += FModel
        #macs, imgt_pred, mask_ratio, _, _, _ = model.get_dynamic_MACs(I0, I1, I2, embt, scale_factor=scale_factor)
        #MACs_list.append(macs.cpu().data)

        mask_ratios[0].extend(mask_ratio[0])
        mask_ratios[1].extend(mask_ratio[1])
        mask_ratios[2].extend(mask_ratio[2])

        #imgt_pred = padder.unpad(imgt_pred)
        psnr = calculate_psnr(imgt_pred, I1).detach().cpu().numpy()
        ssim = calculate_ssim(imgt_pred, I1).detach().cpu().numpy()

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print(
        'eval macs:{:.1f} psnr:{:.3f} ssim:{:.3f} ratio4:{:.4f} ratio3:{:.4f} ratio2:{:.4f}'.format(np.array(MACs_list).mean(), np.array(psnr_list).mean(), np.array(ssim_list).mean(),
                            np.array(mask_ratios[0]).mean(), np.array(mask_ratios[1]).mean(), np.array(mask_ratios[2]).mean()))
    print(F_sum/len(file_list))
    elem_counter = Counter(sizes)
    print(elem_counter)
=======
    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

# Replace the 'path' with your SNU-FILM dataset absolute path.
path = '/home/ltkong/Datasets/SNU-FILM/'

psnr_list = []
ssim_list = []
file_list = []
test_file = "test-hard.txt" # test-easy.txt, test-medium.txt, test-hard.txt, test-extreme.txt
with open(os.path.join(path, test_file), "r") as f:
    for line in f:
        line = line.strip()
        file_list.append(line.split(' '))

for line in file_list:
    print(os.path.join(path, line[0]))
    I0_path = os.path.join(path, line[0])
    I1_path = os.path.join(path, line[1])
    I2_path = os.path.join(path, line[2])
    I0 = read(I0_path)
    I1 = read(I1_path)
    I2 = read(I2_path)
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    padder = InputPadder(I0.shape)
    I0, I2 = padder.pad(I0, I2)
    embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

    I1_pred = model.inference(I0, I2, embt, scale_factor=scale_factor)
    I1_pred = padder.unpad(I1_pred)

    psnr = calculate_psnr(I1_pred, I1).detach().cpu().numpy()
    ssim = calculate_ssim(I1_pred, I1).detach().cpu().numpy()

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    
    print('Avg PSNR: {} SSIM: {}'.format(np.mean(psnr_list), np.mean(ssim_list)))
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
