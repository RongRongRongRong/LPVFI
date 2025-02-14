import sys
import tqdm
import torch
import argparse
import numpy as np
import os.path as osp
import math
from omegaconf import OmegaConf
import os
import sys

sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from collections import Counter
from skimage.color import rgb2yuv, yuv2rgb
from yuv_frame_io import YUV_Read
sys.path.append('.')
from utils.utils import read, img2tensor
from utils.build_utils import build_from_cfg
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

parser = argparse.ArgumentParser(
                prog = 'LPVFI-AMT',
                description = 'HD evaluation',
                )
parser.add_argument('-c', '--config', default='./cfgs/LPVFI-AMT-S.yaml')
parser.add_argument('-p', '--ckpt', default='./weights/LPVFI-AMT.pth')
parser.add_argument('-r', '--root', default='/home/luolab/xzh/IFRNet-main/data/HD_dataset')
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
name_list = [
    #(args.root+'/HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280),
    #(args.root+'/HD720p_GT/shields_1280x720_60.yuv', 720, 1280),
    #(args.root+'/HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280),
    #(args.root+'/HD1080p_GT/BlueSky.yuv', 1080, 1920),
    #(args.root+'/HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920),
    #(args.root+'/HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920),
    #(args.root+'/HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920),
    (args.root+'/HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280),
    (args.root+'/HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280),
    (args.root+'/HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280),
    (args.root+'/HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280),
]
tot = 0.
F_sum = 0
sizes = []
for data in name_list:
    psnr_list = []
    name = data[0]
    h = data[1]
    w = data[2]
    if 'yuv' in name:
        Reader = YUV_Read(name, h, w, toRGB=True)
    else:
        Reader = cv2.VideoCapture(name)
    _, lastframe = Reader.read()
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video = cv2.VideoWriter(name + '.mp4', fourcc, 30, (w, h))
    for index in range(0, 100, 2):
        if 'yuv' in name:
            IMAGE1, success1 = Reader.read(index)
            gt, _ = Reader.read(index + 1)
            IMAGE2, success2 = Reader.read(index + 2)
            if not success2:
                break
        else:
            success1, gt = Reader.read()
            success2, frame = Reader.read()
            IMAGE1 = lastframe
            IMAGE2 = frame
            lastframe = frame
            if not success2:
                break
        I0 = torch.from_numpy(np.transpose(IMAGE1, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
        I1 = torch.from_numpy(np.transpose(IMAGE2, (2,0,1)).astype("float32") / 255.).cuda().unsqueeze(0)
        embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(device)
        if h == 720:
            pad = 24
            scale_factor = 1
        elif h == 1080:
            pad = 0
            scale_factor = 0.8
        else:
            pad = 16
            scale_factor = 1
        pader = torch.nn.ReplicationPad2d([0, 0, pad, pad])
        I0 = pader(I0)
        I1 = pader(I1)
        sizes.append(I0.shape)
        with torch.no_grad():
            pred, FModel = model.get_dynamic_macs(I0, I1, embt,
                            scale_factor=scale_factor, thres=args.thres)
            F_sum += FModel
            if pad!=0:
                pred = pred[:, :, pad: -pad]
        out = (np.round(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')
        if 'yuv' in name:
            diff_rgb = 128.0 + rgb2yuv(gt / 255.)[:, :, 0] * 255 - rgb2yuv(out / 255.)[:, :, 0] * 255
            mse = np.mean((diff_rgb - 128.0) ** 2)
            PIXEL_MAX = 255.0
            psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        else:
            psnr = calculate_psnr(gt, out).detach().cpu().numpy()
        psnr_list.append(psnr)
    print(np.mean(psnr_list))
    tot += np.mean(psnr_list)
print('avg psnr', tot / len(name_list))
print(F_sum/len(sizes))
elem_counter = Counter(sizes)
print(elem_counter)
