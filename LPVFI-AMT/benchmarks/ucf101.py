import os
import sys
import tqdm
import torch
import argparse
import numpy as np
import os.path as osp
from omegaconf import OmegaConf

sys.path.append('.')
from utils.utils import read, img2tensor
from utils.build_utils import build_from_cfg
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

parser = argparse.ArgumentParser(
                prog = 'AMT',
                description = 'UCF101 evaluation',
                )
parser.add_argument('-c', '--config', default='../cfgs/LPVFI-AMT-S.yaml')
parser.add_argument('-p', '--ckpt', default='../weights/.pth')
parser.add_argument('-r', '--root', default='/home/luolab/xzh/IFRNet-main/data/ucf101')
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

dirs = sorted(os.listdir(root))
thres_scales = [5,6,10,15]
for i in range(4):
    psnr_list = []
    ssim_list = []
    F_sum = 0
    pbar = tqdm.tqdm(dirs, total=len(dirs))
    for d in pbar:
        dir_path = osp.join(root, d)
        I0 = img2tensor(read(osp.join(dir_path, 'frame_00.png'))).to(device)
        I1 = img2tensor(read(osp.join(dir_path, 'frame_01_gt.png'))).to(device)
        I2 = img2tensor(read(osp.join(dir_path, 'frame_02.png'))).to(device)
        embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

        with torch.no_grad():
            I1_pred, F_model = model.get_dynamic_macs(I0, I2, embt, thres_scales[i],
                                scale_factor=1.0, thres=args.thres)
        F_sum += F_model/len(pbar)

        psnr = calculate_psnr(I1_pred, I1)
        ssim = calculate_ssim(I1_pred, I1)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        desc_str = f'[{network_name}/UCF101] psnr: {avg_psnr:.02f}, ssim: {avg_ssim:.04f}'
        pbar.set_description_str(desc_str)
    print("F_sum:",F_sum)