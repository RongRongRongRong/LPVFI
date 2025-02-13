from symbol import parameters

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from utils import warp, get_robust_weight
from loss import *
import math



def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.PReLU(out_channels)
    )

def get_forward_mask(mask):
    padh = math.ceil(mask.shape[2] / 8) * 8 - mask.shape[2]
    padw = math.ceil(mask.shape[3] / 8) * 8 - mask.shape[3]
    mask = F.pad(mask, [0, padw, 0, padh])
    return F.interpolate(F.max_pool2d(mask, 8, 8, 0),scale_factor=8,mode='nearest')[:,:,:-padh or None,:-padw or None]


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)

        res_feat = out[:, :-self.side_channels, ...]
        side_feat = out[:, -self.side_channels:, :, :]
        side_feat = self.conv2(side_feat)
        out = self.conv3(torch.cat([res_feat, side_feat], 1))

        res_feat = out[:, :-self.side_channels, ...]
        side_feat = out[:, -self.side_channels:, :, :]
        side_feat = self.conv4(side_feat)
        out = self.conv5(torch.cat([res_feat, side_feat], 1))

        out = self.prelu(x + out)
        return out

    def forward_wmask(self, x, masks):
        out = self.conv1(x*get_forward_mask(masks[0]))
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :]*get_forward_mask(masks[1]))
        out = self.conv3(out*get_forward_mask(masks[2]))
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :]*get_forward_mask(masks[3]))
        out = self.prelu(x + self.conv5(out*get_forward_mask(masks[4])))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, 24, 3, 2, 1),
            convrelu(24, 24, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(24, 36, 3, 2, 1),
            convrelu(36, 36, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(36, 54, 3, 2, 1),
            convrelu(54, 54, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(54, 72, 3, 2, 1),
            convrelu(72, 72, 3, 1, 1)
        )

    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(144+1, 144),
            ResBlock(144, 24),
            nn.ConvTranspose2d(144, 58, 4, 2, 1, bias=True)
        )

    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(166, 162),
            ResBlock(162, 24),
            nn.ConvTranspose2d(162, 40, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out

    def forward_wmask(self, ft_, f0, f1, up_flow0, up_flow1, conv_mask):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        conv_mask_l1 = F.max_pool2d(conv_mask, 4, 2, 1)
        conv_mask_l2 = F.max_pool2d(conv_mask_l1, 3, 1, 1)
        conv_mask_l3 = F.max_pool2d(conv_mask_l2, 3, 1, 1)
        conv_mask_l4 = F.max_pool2d(conv_mask_l3, 3, 1, 1)
        conv_mask_l5 = F.max_pool2d(conv_mask_l4, 3, 1, 1)
        conv_mask_l6 = F.max_pool2d(conv_mask_l5, 3, 1, 1)
        conv_mask_l7 = F.max_pool2d(conv_mask_l6, 3, 1, 1)
        out = self.convblock[0](f_in*get_forward_mask(conv_mask_l7))
        out = self.convblock[1].forward_wmask(out, [conv_mask_l6, conv_mask_l5, conv_mask_l4, conv_mask_l3, conv_mask_l2])
        out = self.convblock[2](out*get_forward_mask(conv_mask_l1))
        return out * conv_mask


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(112, 108),
            ResBlock(108, 24),
            nn.ConvTranspose2d(108, 28, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out

    def forward_wmask(self, ft_, f0, f1, up_flow0, up_flow1, conv_mask):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        conv_mask_l1 = F.max_pool2d(conv_mask, 4, 2, 1)
        conv_mask_l2 = F.max_pool2d(conv_mask_l1, 3, 1, 1)
        conv_mask_l3 = F.max_pool2d(conv_mask_l2, 3, 1, 1)
        conv_mask_l4 = F.max_pool2d(conv_mask_l3, 3, 1, 1)
        conv_mask_l5 = F.max_pool2d(conv_mask_l4, 3, 1, 1)
        conv_mask_l6 = F.max_pool2d(conv_mask_l5, 3, 1, 1)
        conv_mask_l7 = F.max_pool2d(conv_mask_l6, 3, 1, 1)
        out = self.convblock[0](f_in*get_forward_mask(conv_mask_l7))
        out = self.convblock[1].forward_wmask(out, [conv_mask_l6, conv_mask_l5, conv_mask_l4, conv_mask_l3, conv_mask_l2])
        out = self.convblock[2](out*get_forward_mask(conv_mask_l1))
        return out * conv_mask


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(76, 72),
            ResBlock(72, 24),
            nn.ConvTranspose2d(72, 8, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out

    def forward_wmask(self, ft_, f0, f1, up_flow0, up_flow1, conv_mask):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        conv_mask_l1 = F.max_pool2d(conv_mask, 4, 2, 1)
        conv_mask_l2 = F.max_pool2d(conv_mask_l1, 3, 1, 1)
        conv_mask_l3 = F.max_pool2d(conv_mask_l2, 3, 1, 1)
        conv_mask_l4 = F.max_pool2d(conv_mask_l3, 3, 1, 1)
        conv_mask_l5 = F.max_pool2d(conv_mask_l4, 3, 1, 1)
        conv_mask_l6 = F.max_pool2d(conv_mask_l5, 3, 1, 1)
        conv_mask_l7 = F.max_pool2d(conv_mask_l6, 3, 1, 1)
        out = self.convblock[0](f_in*get_forward_mask(conv_mask_l7))
        out = self.convblock[1].forward_wmask(out, [conv_mask_l6, conv_mask_l5, conv_mask_l4, conv_mask_l3, conv_mask_l2])
        out = self.convblock[2](out*get_forward_mask(conv_mask_l1))
        return out * conv_mask


class Model(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(Model, self).__init__()
        self.thres_scale = 4
        self.train_resolution = 224
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)
        self.l1flow_weight = 4e-4
        self.one_count = 0
        self.zero_count = 0



    def test_mask(self, img0, imgt, img1, embt, scale_factor=1.0):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_

        img0_ = resize(img0, scale_factor=scale_factor)
        img1_ = resize(img1, scale_factor=scale_factor)

        mask_ratio = [[],[],[]]
        '''
        need_pad = False
        if img0_.shape[2] % 16 != 0 or img0_.shape[3] % 16 != 0:
            need_pad = True
        if need_pad:
            pad_h = (((16 - img0_.shape[2] % 16) // 2) % 8, ((16 - (img0_.shape[2] % 16) + 1) // 2) % 8)
            pad_w = (((16 - img0_.shape[3] % 16) // 2) % 8, ((16 - (img0_.shape[3] % 16) + 1) // 2) % 8)
            img0_ = F.pad(img0_, (*pad_w, *pad_h))
            img1_ = F.pad(img1_, (*pad_w, *pad_h))
        '''
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0 = out4[:, 0:2]
        up_flow1 = out4[:, 2:4]
        up_mask_4 = out4[:, 4:5]
        ft_d_ = out4[:, 4:]
        thres_scale = 6
        conv_thres = F.adaptive_max_pool2d(torch.abs(out4[:, 0:4]), (1, 1)) / thres_scale
        conv_mask = ((torch.abs(out4[:, 0:1]) < conv_thres[:, 0:1] / 8) & (
                torch.abs(out4[:, 1:2]) < conv_thres[:, 1:2] / 8) & (torch.abs(out4[:, 2:3]) < conv_thres[:, 2:3] / 8)
                     & (torch.abs(out4[:, 3:4]) < conv_thres[:, 3:4] / 8)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2, mode='nearest')
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=15, stride=1, padding=7) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        for i in range(len(conv_mask)):
            mask_ratio[2].append((torch.sum(conv_mask[i]) / conv_mask.shape[2] / conv_mask.shape[3]).cpu().numpy())

        out3 = self.decoder3.forward_wmask(ft_d_, f0_3, f1_3, up_flow0, up_flow1, conv_mask)
        up_flow0 = out3[:, 0:2] + 2.0 * resize(up_flow0, scale_factor=2.0)
        up_flow1 = out3[:, 2:4] + 2.0 * resize(up_flow1, scale_factor=2.0)
        up_mask_3 = out3[:, 4:5] * conv_mask + F.interpolate(up_mask_4, scale_factor=2, mode='nearest')
        ft_d_ = out3[:, 4:]

        conv_thres = F.adaptive_max_pool2d(torch.abs(out3[:, 0:4]), (1, 1)) / thres_scale
        conv_mask = ((torch.abs(out3[:, 0:1]) * conv_mask < conv_thres[:, 0:1] / 4) & (
                torch.abs(out3[:, 1:2]) * conv_mask < conv_thres[:, 1:2] / 4) & (
                                 torch.abs(out3[:, 2:3]) * conv_mask < conv_thres[:, 2:3] / 4)
                     & (torch.abs(out3[:, 3:4]) * conv_mask < conv_thres[:, 3:4] / 4)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2, mode='nearest')
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=15, stride=1, padding=7) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        for i in range(len(conv_mask)):
            mask_ratio[2].append((torch.sum(conv_mask[i]) / conv_mask.shape[2] / conv_mask.shape[3]).cpu().numpy())

        out2 = self.decoder2.forward_wmask(ft_d_, f0_2, f1_2, up_flow0, up_flow1, conv_mask)
        up_flow0 = out2[:, 0:2] + 2.0 * resize(up_flow0, scale_factor=2.0)
        up_flow1 = out2[:, 2:4] + 2.0 * resize(up_flow1, scale_factor=2.0)
        ft_d_ = out2[:, 4:]
        up_mask_2 = out2[:, 4:5] * conv_mask + F.interpolate(up_mask_3, scale_factor=2, mode='nearest')
        conv_thres = F.adaptive_max_pool2d(torch.abs(out2[:, 0:4]), (1, 1)) / thres_scale

        conv_mask = ((torch.abs(out2[:, 0:1]) * conv_mask < conv_thres[:, 0:1] / 2) & (
                torch.abs(out2[:, 1:2]) * conv_mask < conv_thres[:, 1:2] / 2) & (
                                 torch.abs(out2[:, 2:3]) * conv_mask < conv_thres[:, 2:3] / 2)
                     & (torch.abs(out2[:, 3:4]) * conv_mask < conv_thres[:, 3:4] / 2)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2, mode='nearest')
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=15, stride=1, padding=7) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        for i in range(len(conv_mask)):
            mask_ratio[2].append((torch.sum(conv_mask[i]) / conv_mask.shape[2] / conv_mask.shape[3]).cpu().numpy())

        out1 = self.decoder1.forward_wmask(ft_d_, f0_1, f1_1, up_flow0, up_flow1, conv_mask)
        up_flow0 = out1[:, 0:2] + 2.0 * resize(up_flow0, scale_factor=2.0)
        up_flow1 = out1[:, 2:4] + 2.0 * resize(up_flow1, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5] * conv_mask + F.interpolate(up_mask_2, scale_factor=2, mode='nearest'))
        up_res_1 = out1[:, 5:]
        up_flow0 = resize(up_flow0, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
        up_flow1 = resize(up_flow1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
        up_mask_1 = resize(up_mask_1, scale_factor=(1.0 / scale_factor))
        up_res_1 = resize(up_res_1, scale_factor=(1.0 / scale_factor))
        img0_warp = warp(img0, up_flow0)
        img1_warp = warp(img1, up_flow1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)
        return imgt_pred, mask_ratio, up_flow0, up_flow1, conv_mask


    def forward_2(self, img0, img1, embt, imgt, flow):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        loss_l1flow = 0

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        up_mask_4 = out4[:, 4:5]
        ft_d_ = out4[:, 4:]

        conv_thres = F.adaptive_max_pool2d(torch.abs(out4[:, 0:4]),(1,1))/self.thres_scale
        conv_mask = ((torch.abs(out4[:, 0:1]) < conv_thres[:, 0:1] / 8) & (
                torch.abs(out4[:, 1:2]) < conv_thres[:, 1:2] / 8) & (torch.abs(out4[:, 2:3]) < conv_thres[:, 2:3] / 8)
                     & (torch.abs(out4[:, 3:4]) < conv_thres[:, 3:4] / 8)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2)

        #conv_mask_noise = (F.avg_pool2d(conv_mask,kernel_size=11,stride=1,padding=5)>0.2).float()
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=15, stride=1, padding=7) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask


        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        loss_l1flow += F.l1_loss(out4[:, 0:4], torch.zeros(out4[:, 0:4].shape, device='cuda')) * self.l1flow_weight

        out3 = self.decoder3(ft_d_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] * conv_mask + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] * conv_mask + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        up_mask_3 = out3[:, 4:5] * conv_mask + F.interpolate(up_mask_4, scale_factor=2, mode='nearest')
        ft_d_ = out3[:, 4:] * conv_mask

        conv_thres = F.adaptive_max_pool2d(torch.abs(out3[:, 0:4]),(1,1))/self.thres_scale
        conv_mask = ((torch.abs(out3[:, 0:1]) * conv_mask < conv_thres[:, 0:1] / 4) & (
                torch.abs(out3[:, 1:2]) * conv_mask < conv_thres[:, 1:2] / 4) & (torch.abs(out3[:, 2:3]) * conv_mask < conv_thres[:, 2:3] / 4)
                     & (torch.abs(out3[:, 3:4]) * conv_mask< conv_thres[:, 3:4]  / 4)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2)

        #conv_mask_noise = (F.avg_pool2d(conv_mask,kernel_size=21,stride=1,padding=10)>0.2).float()
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=29, stride=1, padding=14) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask


        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        loss_l1flow += F.l1_loss(out3[:, 0:4], torch.zeros(out3[:, 0:4].shape, device='cuda')) * self.l1flow_weight

        out2 = self.decoder2(ft_d_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] * conv_mask + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] * conv_mask + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_d_ = out2[:, 4:] * conv_mask
        up_mask_2 = out2[:, 4:5] * conv_mask + F.interpolate(up_mask_3, scale_factor=2, mode='nearest')

        conv_thres = F.adaptive_max_pool2d(torch.abs(out2[:, 0:4]),(1,1))/self.thres_scale*0.5
        conv_mask = ((torch.abs(out2[:, 0:1]) * conv_mask < conv_thres[:, 0:1] / 2) & (
                torch.abs(out2[:, 1:2]) * conv_mask < conv_thres[:, 1:2] / 2) & (torch.abs(out2[:, 2:3]) * conv_mask < conv_thres[:, 2:3] / 2)
                     & (torch.abs(out2[:, 3:4]) * conv_mask < conv_thres[:, 3:4] / 2)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2)

        #conv_mask_noise = (F.avg_pool2d(conv_mask,kernel_size=41,stride=1,padding=20)>0.2).float()
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=57, stride=1, padding=28) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask

        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        loss_l1flow += F.l1_loss(out2[:, 0:4], torch.zeros(out2[:, 0:4].shape, device='cuda')) * self.l1flow_weight

        out1 = self.decoder1(ft_d_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] * conv_mask + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] * conv_mask + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5] * conv_mask + F.interpolate(up_mask_2, scale_factor=2, mode='nearest'))
        up_res_1 = out1[:, 5:] * conv_mask
        loss_l1flow += F.l1_loss(out1[:, 0:4], torch.zeros(out1[:, 0:4].shape, device='cuda')) * self.l1flow_weight

        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
        empty2 = torch.tensor(0.0, device='cuda')

        return imgt_pred, loss_rec, empty2, empty2, empty2, loss_l1flow, empty2, empty2, empty2


    def forward(self, img0, img1, embt, imgt, flow):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        #imgt_ = imgt - mean_

        loss_l1flow = 0

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1)
        #ft_1, ft_2, ft_3, ft_4 = self.encoder(imgt_)

        dfs = []
        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        up_mask_4 = out4[:, 4:5]
        ft_3_ = out4[:, 4:]
        loss_l1flow += F.l1_loss(out4[:, 0:4], torch.zeros(out4[:, 0:4].shape, device='cuda')) * self.l1flow_weight
        dfs.append((out4[:, 0:2]*8, out4[:, 2:4]*8, ft_3_))

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        up_mask_3 = out3[:, 4:5] + F.interpolate(up_mask_4, scale_factor=2, mode='nearest')
        ft_2_ = out3[:, 4:]
        loss_l1flow += F.l1_loss(out3[:, 0:4], torch.zeros(out3[:, 0:4].shape, device='cuda')) * self.l1flow_weight
        dfs.append((out3[:, 0:2]*4, out3[:, 2:4]*4, ft_2_))

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:]
        up_mask_2 = out2[:, 4:5] + F.interpolate(up_mask_3, scale_factor=2, mode='nearest')
        loss_l1flow += F.l1_loss(out2[:, 0:4], torch.zeros(out2[:, 0:4].shape, device='cuda')) * self.l1flow_weight
        dfs.append((out2[:, 0:2]*2, out2[:, 2:4]*2, ft_1_))

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5] + F.interpolate(up_mask_2, scale_factor=2, mode='nearest'))
        up_res_1 = out1[:, 5:]
        ft_0_ = out1[:,4:]
        loss_l1flow += F.l1_loss(out1[:, 0:4], torch.zeros(out1[:, 0:4].shape, device='cuda')) * self.l1flow_weight
        dfs.append((out1[:, 0:2], out1[:, 2:4], ft_0_))

        img0_warp = warp(img0, up_flow0_1)
        img1_warp = warp(img1, up_flow1_1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
        #loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3))
        loss_geo = torch.tensor(0.0,device='cuda')

        loss_dft = 0
        loss_sflow = 0
        loss_rflow = 0
        rflow_weight = 7e-5
        dft_weight = 1
        sflow_weight = 0.007
        loss_dflow = 0

        if flow is not None:
            robust_weight0 = get_robust_weight(up_flow0_1, flow[:, 0:2], beta=0.3)
            robust_weight1 = get_robust_weight(up_flow1_1, flow[:, 2:4], beta=0.3)
            loss_dis = 0.01 * (self.rb_loss(2.0 * resize(up_flow0_2, 2.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(2.0 * resize(up_flow1_2, 2.0) - flow[:, 2:4], weight=robust_weight1))
            loss_dis += 0.01 * (self.rb_loss(4.0 * resize(up_flow0_3, 4.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(4.0 * resize(up_flow1_3, 4.0) - flow[:, 2:4], weight=robust_weight1))
            loss_dis += 0.01 * (self.rb_loss(8.0 * resize(up_flow0_4, 8.0) - flow[:, 0:2], weight=robust_weight0) + self.rb_loss(8.0 * resize(up_flow1_4, 8.0) - flow[:, 2:4], weight=robust_weight1))
        else:
            loss_dis = 0.00

        # loss rflow
        for i in range(len(dfs)-1):
            flow_avg0 = F.avg_pool2d(dfs[i+1][0],2,2)
            flow_avg1 = F.avg_pool2d(dfs[i+1][1],2,2)
            robust_weight0 = get_robust_weight(dfs[i+1][0], F.interpolate(flow_avg0,scale_factor=2.0), beta=10.0)
            robust_weight1 = get_robust_weight(dfs[i+1][1], F.interpolate(flow_avg1,scale_factor=2.0), beta=10.0)
            robust_weight0 = -F.max_pool2d(-robust_weight0, 2, 2)
            robust_weight1 = -F.max_pool2d(-robust_weight1, 2, 2)
            loss_rflow += ((((dfs[i][0]+flow_avg0).detach() - dfs[i][0]) ** 2).mean(1, True) ** 0.5 * robust_weight0.detach()).mean()
            loss_rflow += ((((dfs[i][1]+flow_avg1).detach() - dfs[i][1]) ** 2).mean(1, True) ** 0.5 * robust_weight1.detach()).mean()
            loss_rflow += ((((dfs[i+1][0]-F.interpolate(flow_avg0,scale_factor=2.0)).detach() - dfs[i+1][0]) ** 2).mean(1, True) ** 0.5 * F.interpolate(robust_weight0,scale_factor=2.0).detach()).mean()
            loss_rflow += ((((dfs[i+1][1]-F.interpolate(flow_avg1,scale_factor=2.0)).detach() - dfs[i+1][1]) ** 2).mean(1, True) ** 0.5 * F.interpolate(robust_weight1,scale_factor=2.0).detach()).mean()
        loss_rflow = loss_rflow * rflow_weight

        for i in range(len(dfs) - 1):
            scale_dif1 = torch.max(torch.zeros(dfs[i + 1][0].shape, device='cuda'),
                      torch.abs(dfs[i + 1][0]) - F.interpolate(torch.abs(dfs[i][0]),
                                                               scale_factor=2))

            scale_dif2 = torch.max(torch.zeros(dfs[i + 1][1].shape, device='cuda'),
                      torch.abs(dfs[i + 1][1]) - F.interpolate(torch.abs(dfs[i][1]),
                                                               scale_factor=2))
            loss_sflow += sflow_weight * F.l1_loss(scale_dif2,torch.zeros(scale_dif2.shape, device='cuda'))

            loss_sflow += sflow_weight * F.l1_loss(scale_dif1,
                      torch.zeros(scale_dif1.shape, device='cuda'))

        for i in range(len(dfs)):
            flow_norm = torch.norm(torch.cat([dfs[i][0],dfs[i][1]],dim=1),p=1,dim=1)
            flow_norm = F.normalize(flow_norm, dim=(1, 2), p=1)
            ft_norm = torch.norm(dfs[i][2],p=1,dim=1)
            ft_norm = F.normalize(ft_norm, dim=(1, 2), p=1)
            loss_dft += dft_weight * F.l1_loss(flow_norm, ft_norm)

        return imgt_pred, loss_rec, loss_geo, loss_rflow, loss_sflow, loss_l1flow, loss_dis, loss_dft, loss_dflow


    def get_dynamic_MACs(self, img0, imgt, img1, embt, scale_factor=1.0, thres=15):
        conv_masks = []
        #get masks
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        img0_ = resize(img0, scale_factor=scale_factor)
        img1_ = resize(img1, scale_factor=scale_factor)
        H, W = img0_.shape[2], img0_.shape[3]

        mask_ratio = [[],[],[]]
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0 = out4[:, 0:2]
        up_flow1 = out4[:, 2:4]
        up_mask_4 = out4[:, 4:5]
        ft_d_ = out4[:, 4:]

        thres_scale = thres
        conv_thres = F.adaptive_max_pool2d(torch.abs(out4[:, 0:4]),(1,1))/thres_scale
        conv_mask = ((torch.abs(out4[:, 0:1]) < conv_thres[:, 0:1] / 8) & (
                torch.abs(out4[:, 1:2]) < conv_thres[:, 1:2] / 8) & (torch.abs(out4[:, 2:3]) < conv_thres[:, 2:3] / 8)
                     & (torch.abs(out4[:, 3:4]) < conv_thres[:, 3:4] / 8)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2, mode='nearest')
        conv_mask_noise = (F.avg_pool2d(conv_mask,kernel_size=15,stride=1,padding=7)>0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        conv_masks.append(conv_mask)
        for i in range(len(conv_mask)):
            mask_ratio[2].append((torch.sum(conv_mask[i]) / conv_mask.shape[2] / conv_mask.shape[3]).cpu().numpy())


        out3 = self.decoder3.forward_wmask(ft_d_, f0_3, f1_3, up_flow0, up_flow1, conv_mask)
        up_flow0 = out3[:, 0:2] + 2.0 * resize(up_flow0, scale_factor=2.0)
        up_flow1 = out3[:, 2:4] + 2.0 * resize(up_flow1, scale_factor=2.0)
        up_mask_3 = out3[:, 4:5] * conv_mask + F.interpolate(up_mask_4, scale_factor=2, mode='nearest')
        ft_d_ = out3[:, 4:]

        conv_thres = F.adaptive_max_pool2d(torch.abs(out3[:, 0:4]),(1,1))/thres_scale
        conv_mask = ((torch.abs(out3[:, 0:1]) * conv_mask < conv_thres[:, 0:1] / 4) & (
                torch.abs(out3[:, 1:2]) * conv_mask < conv_thres[:, 1:2] / 4) & (torch.abs(out3[:, 2:3]) * conv_mask < conv_thres[:, 2:3] / 4)
                     & (torch.abs(out3[:, 3:4]) * conv_mask< conv_thres[:, 3:4]  / 4)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2, mode='nearest')
        conv_mask_noise = (F.avg_pool2d(conv_mask,kernel_size=15,stride=1,padding=7)>0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        conv_masks.append(conv_mask)
        for i in range(len(conv_mask)):
            mask_ratio[2].append((torch.sum(conv_mask[i]) / conv_mask.shape[2] / conv_mask.shape[3]).cpu().numpy())


        out2 = self.decoder2.forward_wmask(ft_d_, f0_2, f1_2, up_flow0, up_flow1, conv_mask)
        up_flow0 = out2[:, 0:2] + 2.0 * resize(up_flow0, scale_factor=2.0)
        up_flow1 = out2[:, 2:4] + 2.0 * resize(up_flow1, scale_factor=2.0)
        ft_d_ = out2[:, 4:]
        up_mask_2 = out2[:, 4:5] * conv_mask + F.interpolate(up_mask_3, scale_factor=2, mode='nearest')
        conv_thres = F.adaptive_max_pool2d(torch.abs(out2[:, 0:4]),(1,1))/thres_scale

        conv_mask = ((torch.abs(out2[:, 0:1]) * conv_mask < conv_thres[:, 0:1] / 2) & (
                torch.abs(out2[:, 1:2]) * conv_mask < conv_thres[:, 1:2] / 2) & (torch.abs(out2[:, 2:3]) * conv_mask < conv_thres[:, 2:3] / 2)
                     & (torch.abs(out2[:, 3:4]) * conv_mask < conv_thres[:, 3:4] / 2)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2, mode='nearest')
        conv_mask_noise = (F.avg_pool2d(conv_mask,kernel_size=15,stride=1,padding=7)>0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        conv_masks.append(conv_mask)
        for i in range(len(conv_mask)):
            mask_ratio[2].append((torch.sum(conv_mask[i]) / conv_mask.shape[2] / conv_mask.shape[3]).cpu().numpy())


        out1 = self.decoder1.forward_wmask(ft_d_, f0_1, f1_1, up_flow0, up_flow1, conv_mask)
        up_flow0 = out1[:, 0:2] + 2.0 * resize(up_flow0, scale_factor=2.0)
        up_flow1 = out1[:, 2:4] + 2.0 * resize(up_flow1, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5] * conv_mask + F.interpolate(up_mask_2, scale_factor=2, mode='nearest'))
        up_res_1 = out1[:, 5:]
        up_flow0 = resize(up_flow0, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
        up_flow1 = resize(up_flow1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
        up_mask_1 = resize(up_mask_1, scale_factor=(1.0 / scale_factor))
        up_res_1 = resize(up_res_1, scale_factor=(1.0 / scale_factor))
        img0_warp = warp(img0, up_flow0)
        img1_warp = warp(img1, up_flow1)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_
        imgt_pred = imgt_merge + up_res_1
        imgt_pred = torch.clamp(imgt_pred, 0, 1)



        #Encoder FLOPs
        F_encoder = 0
        for n,m in self.encoder.named_modules():
            if type(m) is nn.Conv2d:
                if m.stride[0]==2:
                    H/=2
                    W/=2
                F_encoder += H * W * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9
            elif type(m) is nn.PReLU:
                c_in = m.num_parameters
                F_encoder += c_in * H * W * 1e-9
        F_encoder *= 2  #two images

        #Decoder4 FLOPs
        F_decoder4 = 0
        for n, m in self.decoder4.named_modules():
            if type(m) is nn.Conv2d:
                F_decoder4 += H * W * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9
            elif type(m) is nn.PReLU:
                c_in = m.num_parameters
                F_decoder4 += c_in * H * W * 1e-9
            elif type(m) is nn.ConvTranspose2d: #stride==2
                H *= 2
                W *= 2
                F_decoder4 += H * W * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9

        #Decoder3 FLOPs
        conv_mask = conv_masks[0]
        conv_mask_list = []
        conv_mask_list.append(get_forward_mask(conv_mask))
        conv_mask = F.max_pool2d(conv_mask, 4, 2, 1)
        conv_mask_list.append(get_forward_mask(conv_mask))
        for i in range(5):
            conv_mask =  F.max_pool2d(conv_mask, 3, 1, 1)
            conv_mask_list.append(get_forward_mask(conv_mask))
        conv_mask_index = 6
        F_decoder3 = 0
        for n, m in self.decoder3.named_modules():
            if type(m) is nn.Conv2d:
                F_decoder3 += (torch.sum(conv_mask_list[conv_mask_index])  # (torch.sum(conv_mask_list[conv_mask_index]) * 8 * 8 = H * W?
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
                conv_mask_index -= 1
            elif type(m) is nn.PReLU:
                c_in = m.num_parameters
                F_decoder3 += c_in * H * W * 1e-9
            elif type(m) is nn.ConvTranspose2d: #stride==2
                H *= 2
                W *= 2
                F_decoder3 += (torch.sum(conv_mask_list[conv_mask_index])
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)


        # Decoder2 FLOPs
        conv_mask = conv_masks[1]
        conv_mask_list = []
        conv_mask_list.append(get_forward_mask(conv_mask))
        conv_mask = F.max_pool2d(conv_mask, 4, 2, 1)
        conv_mask_list.append(get_forward_mask(conv_mask))
        for i in range(5):
            conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
            conv_mask_list.append(get_forward_mask(conv_mask))
        conv_mask_index = 6
        F_decoder2 = 0
        for n, m in self.decoder2.named_modules():
            if type(m) is nn.Conv2d:
                F_decoder2 += (torch.sum(conv_mask_list[conv_mask_index]) # (torch.sum(conv_mask_list[conv_mask_index]) * 8 * 8 = H * W?
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
                conv_mask_index -= 1
            elif type(m) is nn.PReLU:
                c_in = m.num_parameters
                F_decoder2 += c_in * H * W * 1e-9
            elif type(m) is nn.ConvTranspose2d: #stride==2
                H *= 2
                W *= 2
                F_decoder2 += (torch.sum(conv_mask_list[conv_mask_index])
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)


        # Decoder1 FLOPs
        conv_mask = conv_masks[2]
        conv_mask_list = []
        conv_mask_list.append(get_forward_mask(conv_mask))
        conv_mask = F.max_pool2d(conv_mask, 4, 2, 1)
        conv_mask_list.append(get_forward_mask(conv_mask))
        for i in range(5):
            conv_mask =  F.max_pool2d(conv_mask, 3, 1, 1)
            conv_mask_list.append(get_forward_mask(conv_mask))
        conv_mask_index = 6
        F_decoder1 = 0
        for n, m in self.decoder1.named_modules():
            if type(m) is nn.Conv2d:
                F_decoder1 += (torch.sum(conv_mask_list[conv_mask_index])  # (torch.sum(conv_mask_list[conv_mask_index]) * 8 * 8 = H * W?
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
                conv_mask_index -= 1
            elif type(m) is nn.PReLU:
                c_in = m.num_parameters
                F_decoder1 += c_in * H * W * 1e-9
            elif type(m) is nn.ConvTranspose2d: #stride==2
                H *= 2
                W *= 2
                F_decoder1 += (torch.sum(conv_mask_list[conv_mask_index])
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)

        F_model = F_encoder + F_decoder4 + F_decoder3 + F_decoder2 + F_decoder1
        return F_model, imgt_pred, mask_ratio, up_flow0, up_flow1, conv_mask