import sys
import tqdm
import torch
import argparse
import numpy as np
import os.path as osp
import math
from omegaconf import OmegaConf

sys.path.append('.')
from utils.utils import read, img2tensor
from utils.build_utils import build_from_cfg
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

parser = argparse.ArgumentParser(
                prog = 'LPVFI-AMT',
                description = 'Vimeo90k evaluation',
                )
parser.add_argument('-c', '--config', default='./cfgs/LPVFI-AMT-S.yaml')
parser.add_argument('-p', '--ckpt', default='./weights/LPVFI-AMT.pth',)
parser.add_argument('-r', '--root', default='/home/luolab/xzh/IFRNet-main/data/Vimeo90k',)
parser.add_argument('--thres', default=15, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg_path = args.config
ckpt_path = args.ckpt
root = args.root

network_cfg = OmegaConf.load(cfg_path).network
network_name = network_cfg.name
model = build_from_cfg(network_cfg)
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['state_dict'])
model = model.to(device)
model.eval()

with open(osp.join(root, 'tri_testlist.txt'), 'r') as fr:
    file_list = fr.readlines()


psnr_list = []
ssim_list = []

pbar = tqdm.tqdm(file_list, total=len(file_list))
F_sum = 0
for name in pbar:
    name = str(name).strip()
    if(len(name) <= 1):
        continue
    dir_path = osp.join(root, 'sequences', name)
    I0 = img2tensor(read(osp.join(dir_path, 'im1.png'))).to(device)
    I1 = img2tensor(read(osp.join(dir_path, 'im2.png'))).to(device)
    I2 = img2tensor(read(osp.join(dir_path, 'im3.png'))).to(device)
    embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

    with torch.no_grad():
        I1_pred, F_model = model.get_dynamic_macs(I0, I2, embt,
                            scale_factor=1.0, thres=args.thres)
    F_sum += F_model/len(file_list)

    psnr = calculate_psnr(I1_pred, I1)
    ssim = calculate_ssim(I1_pred, I1)

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    desc_str = f'[{network_name}/Vimeo90K] psnr: {avg_psnr:.02f}, ssim: {avg_ssim:.04f}'
    pbar.set_description_str(desc_str)
print("F_sum:",F_sum)

