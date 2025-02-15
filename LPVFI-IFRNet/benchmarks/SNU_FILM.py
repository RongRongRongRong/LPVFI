import os
import sys
import tqdm
import argparse
sys.path.append('.')
import torch
import torch.nn.functional as F
import numpy as np
from utils import read
from metric import calculate_psnr, calculate_ssim
from models.LPVFI_S import Model

parser = argparse.ArgumentParser(
                prog = 'LPVFI-IFRNet',
                description = 'SNU-FILM evaluation',
                )
parser.add_argument('-p', '--ckpt', default='./weights/LPVFI-IFRNet.pth')
parser.add_argument('-r', '--root', default='/home/luolab/xzh/IFRNet-main/data/SNU-FILM')
parser.add_argument('--thres', default=15, type=int)
args = parser.parse_args()

# Replace the 'path' with your SNU-FILM dataset absolute path.
path = args.root
test_files = ["test-easy.txt","test-medium.txt","test-hard.txt","test-extreme.txt"]
thres = args.thres
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
model.load_state_dict(torch.load(args.ckpt))
model.eval()
model.cuda()

divisor = 20 #20
scale_factor = 0.8 #0.8

class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor=divisor):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]
    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

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
    for line in file_list:
        #print(os.path.join(prefix_path, line[0]))
        I0_path = os.path.join(args.root, '../..', line[0])
        I1_path = os.path.join(args.root, '../..', line[1])
        I2_path = os.path.join(args.root, '../..', line[2])
        I0 = read(I0_path)
        I1 = read(I1_path)
        I2 = read(I2_path)
        I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
        I1 = (torch.tensor(I1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
        I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
        padder = InputPadder(I0.shape)
        I0, I2 = padder.pad(I0, I2)
        embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)
        with torch.no_grad():
            FModel, imgt_pred, mask_ratio, _, _, _ = model.get_dynamic_MACs(I0, I1, I2, embt, scale_factor=scale_factor, thres=thres)
        F_sum += FModel

        mask_ratios[0].extend(mask_ratio[0])
        mask_ratios[1].extend(mask_ratio[1])
        mask_ratios[2].extend(mask_ratio[2])

        imgt_pred = padder.unpad(imgt_pred)
        psnr = calculate_psnr(imgt_pred, I1).detach().cpu().numpy()
        ssim = calculate_ssim(imgt_pred, I1).detach().cpu().numpy()

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print(test_files+":")
    print(
        'eval macs:{:.1f} psnr:{:.3f} ssim:{:.3f} ratio4:{:.4f} ratio3:{:.4f} ratio2:{:.4f}'.format(F_sum/len(file_list), np.array(psnr_list).mean(), np.array(ssim_list).mean(),
                            np.array(mask_ratios[0]).mean(), np.array(mask_ratios[1]).mean(), np.array(mask_ratios[2]).mean()))
