import os
import math
import time
import cv2
import argparse
import numpy as np
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from datasets import XVFI_Test_Dataset, Vimeo90K_Test_Dataset, UCF_Test_Dataset, SNU_Test_Dataset
from metric import calculate_psnr, calculate_ssim
import logging



def val(args, model):
    os.makedirs(args.log_path, exist_ok=True)
    log_path = os.path.join(args.log_path, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(os.path.join(log_path, 'train.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    logger.info(args)
    if args.dataset == 'Vimeo90k':
        dataset_val = Vimeo90K_Test_Dataset()
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=8, shuffle=False, drop_last=False)
    elif args.dataset == 'UCF':
        dataset_val = UCF_Test_Dataset()
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=8, shuffle=False, drop_last=False)
    evaluate(args, model,dataloader_val, logger)


def evaluate(args, model, dataloader_val, logger):
    psnr_list = []
    ssim_list = []
    model.eval()
    progress_bar = tqdm.tqdm(enumerate(dataloader_val), total=len(dataloader_val))
    Fmodel_sum = 0
    for i, data in progress_bar:
        for l in range(len(data)):
            data[l] = data[l].to(args.device)
        img0, imgt, img1, _, embt = data
        with torch.no_grad():
            Fmodel, imgt_pred, mask_ratio, flow0, flow1, conv_mask = model.get_dynamic_MACs(img0, imgt, img1, embt, scale_factor=1.0, thres=args.thres)
            Fmodel_sum += Fmodel
        for j in range(img0.shape[0]):
            psnr = calculate_psnr(imgt_pred[j].unsqueeze(0), imgt[j].unsqueeze(0)).cpu().data
            ssim = calculate_ssim(imgt_pred[j].unsqueeze(0), imgt[j].unsqueeze(0)).cpu().data
            psnr_list.append(psnr)
            ssim_list.append(ssim)
    print('GMACs:.1f'.format(Fmodel_sum/len(progress_bar)))
    logger.info(
        'eval psnr:{:.2f} ssim:{:.3f}'.format(np.array(psnr_list).mean(), np.array(ssim_list).mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IFRNet')
    parser.add_argument('--model_name', default='My', type=str, help='My, IFRNet_S')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_path', default='./expirements/test_mask', type=str)
    parser.add_argument('--resume_path', default='./weights/', type=str)
    parser.add_argument('--dataset', default='Vimeo90k', type=str, help='Vimeo90k, UCF')
    parser.add_argument('--thres', default=15, type=int)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda')
    if args.model_name == 'My':
        from models.LPVFI_S import Model
        args.log_path = args.log_path + '/' + 'My' + args.dataset
    else:
        from models.IFRNet_S import Model
        args.log_path = args.log_path + '/' + 'baseline_' + args.dataset
    model = Model().to(args.device)
    model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
    val(args, model)