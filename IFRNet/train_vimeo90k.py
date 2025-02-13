import os
import math
import time
import random
import argparse
<<<<<<< HEAD

import cv2
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
=======
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
from datasets import Vimeo90K_Train_Dataset, Vimeo90K_Test_Dataset
from metric import calculate_psnr, calculate_ssim
from utils import AverageMeter
import logging

<<<<<<< HEAD
def random_crop(img0, imgt, img1, flow, crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih-h+1)
    y = np.random.randint(0, iw-w+1)
    img0 = img0[:,:,x:x+h, y:y+w]
    imgt = imgt[:,:,x:x+h, y:y+w]
    img1 = img1[:,:,x:x+h, y:y+w]
    flow = flow[:,:,x:x+h, y:y+w]
    return img0, imgt, img1, flow

def get_lr(args, iters):
    epochs = 300;
    if iters/args.iters_per_epoch>=300:
        lr = args.lr_end
    elif iters<2000:
        lr = args.lr_start * iters / 2000
    else:
        ratio = 0.5 * (1.0 + np.cos((iters-2000) / (epochs * args.iters_per_epoch - 2000) * math.pi))
        lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
=======

def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


<<<<<<< HEAD
def train(args, model):
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

    dataset_train = Vimeo90K_Train_Dataset(augment=True)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    args.iters_per_epoch = dataloader_train.__len__()
    iters = args.resume_epoch * args.iters_per_epoch

    dataset_val = Vimeo90K_Test_Dataset()
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=8, shuffle=False, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr_start, weight_decay=1e-3)
    avg_rec = 0
    avg_sflow = 0
    avg_rflow = 0
    avg_l1flow = 0
    avg_dis = 0
    avg_dft = 0
    avg_geo = 0
    avg_dflow = 0
    best_psnr = 0.0
    for epoch in range(args.resume_epoch, args.epochs):
        if epoch==300:
            dataset_train = Vimeo90K_Train_Dataset(augment=True, crop_size=(256, 384))
            dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                          drop_last=True)
        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for i, data in progress_bar:
            iters += 1
            for l in range(len(data)):
                data[l] = data[l].to(args.device)

            img0, imgt, img1, flow, embt = data
=======
def train(args, ddp_model):
    local_rank = args.local_rank
    print('Distributed Data Parallel Training IFRNet on Rank {}'.format(local_rank))

    if local_rank == 0:
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

    dataset_train = Vimeo90K_Train_Dataset(dataset_dir='/home/ltkong/Datasets/Vimeo90K/vimeo_triplet', augment=True)
    sampler = DistributedSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=sampler)
    args.iters_per_epoch = dataloader_train.__len__()
    iters = args.resume_epoch * args.iters_per_epoch
    
    dataset_val = Vimeo90K_Test_Dataset(dataset_dir='/home/ltkong/Datasets/Vimeo90K/vimeo_triplet')
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=16, pin_memory=True, shuffle=False, drop_last=True)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr_start, weight_decay=0)

    time_stamp = time.time()
    avg_rec = AverageMeter()
    avg_geo = AverageMeter()
    avg_dis = AverageMeter()
    best_psnr = 0.0

    for epoch in range(args.resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader_train):
            for l in range(len(data)):
                data[l] = data[l].to(args.device)
            img0, imgt, img1, flow, embt = data

            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
            lr = get_lr(args, iters)
            set_lr(optimizer, lr)

            optimizer.zero_grad()
<<<<<<< HEAD
            with torch.autograd.set_detect_anomaly(True):
                if epoch<args.switch_epoch:
                    imgt_pred, loss_rec, loss_geo, loss_rflow, loss_sflow, loss_l1flow, loss_dis, loss_dft, loss_dflow = model(img0, img1, embt, imgt, flow)
                else:
                    imgt_pred, loss_rec, loss_geo, loss_rflow, loss_sflow, loss_l1flow, loss_dis, loss_dft, loss_dflow = model.forward_2(img0,
                                                                                                                   img1,
                                                                                                                   embt,
                                                                                                                   imgt,
                                                                                                                   flow)
                loss = loss_rec + loss_dis + loss_sflow + loss_rflow + loss_l1flow + loss_dft + loss_geo + loss_dflow
                loss.backward()
                optimizer.step()
            avg_rec += loss_rec.data
            avg_sflow += loss_sflow.data
            avg_rflow += loss_rflow.data
            avg_l1flow += loss_l1flow.data
            avg_dis += loss_l1flow.data
            avg_dft += loss_dft.data
            avg_dflow += loss_dflow.data
        avg_rec/=len(dataloader_train)
        avg_sflow /= len(dataloader_train)
        avg_rflow /= len(dataloader_train)
        avg_l1flow /= len(dataloader_train)
        avg_dis /= len(dataloader_train)
        avg_dft /= len(dataloader_train)
        avg_dflow /= len(dataloader_train)

        logger.info(
            'epoch:{}/{} loss_rec:{:.4e} loss_geo:{:.4e} loss_sflow:{:.4e} loss_rflow:{:.7e} loss_l1flow:{:.7e} loss_dis:{:.7e} loss_dft:{:.7e} loss:{:.7e}'
                .format(epoch + 1, args.epochs, avg_rec, avg_geo, avg_sflow, avg_rflow, avg_l1flow, avg_dis, avg_dft, avg_dflow))
        avg_rec = 0
        avg_sflow= 0
        avg_rflow= 0
        avg_l1flow= 0
        avg_dis= 0
        avg_dft= 0
        avg_geo= 0
        avg_dflow = 0

        if (epoch+1) % 40==0:
            torch.save(model.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'epoch'+str(epoch)))

        if (epoch+1) % args.eval_interval == 0:
            psnr = evaluate(args, model, dataloader_val, epoch, logger)
            if psnr > best_psnr and epoch>200:
                best_psnr = psnr
                torch.save(model.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'best'))
            torch.save(model.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'latest'))


def evaluate(args, model, dataloader_val, epoch, logger):
    loss_rec_list = []
    loss_sflow_list = []
    loss_rflow_list = []
    loss_l1flow_list = []
    loss_dis_list = []
    loss_dft_list = []
    loss_geo_list = []
    loss_dflow_list = []
    psnr_list = []
    for data in tqdm(dataloader_val, total=len(dataloader_val)):
=======

            imgt_pred, loss_rec, loss_geo, loss_dis = ddp_model(img0, img1, embt, imgt, flow)

            loss = loss_rec + loss_geo + loss_dis
            loss.backward()
            optimizer.step()

            avg_rec.update(loss_rec.cpu().data)
            avg_geo.update(loss_geo.cpu().data)
            avg_dis.update(loss_dis.cpu().data)
            train_time_interval = time.time() - time_stamp

            if (iters+1) % 100 == 0 and local_rank == 0:
                logger.info('epoch:{}/{} iter:{}/{} time:{:.2f}+{:.2f} lr:{:.5e} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e}'.format(epoch+1, args.epochs, iters+1, args.epochs * args.iters_per_epoch, data_time_interval, train_time_interval, lr, avg_rec.avg, avg_geo.avg, avg_dis.avg))
                avg_rec.reset()
                avg_geo.reset()
                avg_dis.reset()

            iters += 1
            time_stamp = time.time()

        if (epoch+1) % args.eval_interval == 0 and local_rank == 0:
            psnr = evaluate(args, ddp_model, dataloader_val, epoch, logger)
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(ddp_model.module.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'best'))
            torch.save(ddp_model.module.state_dict(), '{}/{}_{}.pth'.format(log_path, args.model_name, 'latest'))

        dist.barrier()


def evaluate(args, ddp_model, dataloader_val, epoch, logger):
    loss_rec_list = []
    loss_geo_list = []
    loss_dis_list = []
    psnr_list = []
    time_stamp = time.time()
    for i, data in enumerate(dataloader_val):
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
        for l in range(len(data)):
            data[l] = data[l].to(args.device)
        img0, imgt, img1, flow, embt = data

        with torch.no_grad():
<<<<<<< HEAD
            if epoch<args.switch_epoch:
                imgt_pred, loss_rec, loss_geo, loss_rflow, loss_sflow, loss_l1flow, loss_dis, loss_dft, loss_dflow = model(img0, img1, embt, imgt, flow)
            else:
                imgt_pred, loss_rec, loss_geo, loss_rflow, loss_sflow, loss_l1flow, loss_dis, loss_dft, loss_dflow = model.forward_2(img0,
                                                                                                               img1,
                                                                                                               embt,
                                                                                                               imgt,
                                                                                                               flow)
        loss_rec_list.append(loss_rec.cpu().numpy())
        loss_sflow_list.append(loss_sflow.cpu().numpy())
        loss_rflow_list.append(loss_rflow.cpu().numpy())
        loss_l1flow_list.append(loss_l1flow.cpu().numpy())
        loss_dis_list.append(loss_dis.cpu().numpy())
        loss_dft_list.append(loss_dft.cpu().numpy())
        loss_geo_list.append(loss_geo.cpu().numpy())
        loss_dflow_list.append(loss_dflow.cpu().numpy())
=======
            imgt_pred, loss_rec, loss_geo, loss_dis = ddp_model(img0, img1, embt, imgt, flow)

        loss_rec_list.append(loss_rec.cpu().numpy())
        loss_geo_list.append(loss_geo.cpu().numpy())
        loss_dis_list.append(loss_dis.cpu().numpy())
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe

        for j in range(img0.shape[0]):
            psnr = calculate_psnr(imgt_pred[j].unsqueeze(0), imgt[j].unsqueeze(0)).cpu().data
            psnr_list.append(psnr)

<<<<<<< HEAD
    logger.info('eval epoch:{}/{} loss_rec:{:.4e} loss_geo:{:.4e} loss_sflow:{:.4e} psnr:{:.3f} loss_rflow:{:.7e} loss_l1flow:{:.7e} loss_dis:{:.7e} loss_dft:{:.7e} loss_dflow:{:.7e}'.format(epoch+1,
                   args.epochs, np.array(loss_rec_list).mean(), np.array(loss_geo_list).mean(), np.array(loss_sflow_list).mean(),
                        np.array(psnr_list).mean(), np.array(loss_rflow_list).mean(),
                             np.array(loss_l1flow_list).mean(), np.array(loss_dis_list).mean(), np.array(loss_dft_list).mean(), np.array(loss_dflow_list).mean()))
=======
    eval_time_interval = time.time() - time_stamp
    
    logger.info('eval epoch:{}/{} time:{:.2f} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e} psnr:{:.3f}'.format(epoch+1, args.epochs, eval_time_interval, np.array(loss_rec_list).mean(), np.array(loss_geo_list).mean(), np.array(loss_dis_list).mean(), np.array(psnr_list).mean()))
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
    return np.array(psnr_list).mean()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IFRNet')
<<<<<<< HEAD
    parser.add_argument('--model_name', default='', type=str, help='IFRNet, IFRNet_L, IFRNet_S')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--epochs', default=350, type=int)
    parser.add_argument('--switch_epoch', default=200, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)

    parser.add_argument('--batch_size', default=15, type=int)
    parser.add_argument('--lr_start', default=2e-4, type=float)
    parser.add_argument('--lr_end', default=2e-5, type=float)
    parser.add_argument('--log_path', default='./expirements/checkpoint', type=str)
    parser.add_argument('--resume_epoch', default=300, type=int)
    parser.add_argument('--resume_path', default='./weights/', type=str)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda')
=======
    parser.add_argument('--model_name', default='IFRNet', type=str, help='IFRNet, IFRNet_L, IFRNet_S')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--world_size', default=4, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--lr_start', default=1e-4, type=float)
    parser.add_argument('--lr_end', default=1e-5, type=float)
    parser.add_argument('--log_path', default='checkpoint', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl', world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
<<<<<<< HEAD

    from models.LPVFI_S import Model
    args.log_path = args.log_path + '/' + 'My_V90'
    model = Model().to(args.device)
    model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
    train(args, model)
=======
    torch.backends.cudnn.benchmark = True

    if args.model_name == 'IFRNet':
        from models.IFRNet import Model
    elif args.model_name == 'IFRNet_L':
        from models.IFRNet_L import Model
    elif args.model_name == 'IFRNet_S':
        from models.IFRNet_S import Model

    args.log_path = args.log_path + '/' + args.model_name
    args.num_workers = args.batch_size

    model = Model().to(args.device)
    
    if args.resume_epoch != 0:
        model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
        
    ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    train(args, ddp_model)
    
    dist.destroy_process_group()
>>>>>>> b117bcafcf074b2de756b882f8a6ca02c3169bfe
