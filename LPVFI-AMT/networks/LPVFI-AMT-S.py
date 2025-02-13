import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from networks.blocks.raft import (
    coords_grid,
    SmallUpdateBlock, BidirCorrBlock
)
from networks.blocks.feat_enc import (
    SmallEncoder
)
from networks.blocks.ifrnet import (
    resize,
    Encoder,
    InitDecoder,
    IntermediateDecoder, convrelu
)
from networks.blocks.multi_flow import (
    multi_flow_combine,
    MultiFlowDecoder
)

def get_robust_weight(flow_pred, flow_gt, beta):
    epe = ((flow_pred.detach() - flow_gt) ** 2).sum(dim=1, keepdim=True) ** 0.5
    robust_weight = torch.exp(-beta * epe)
    return robust_weight

def get_forward_mask(mask):
    padh = math.ceil(mask.shape[2] / 8) * 8 - mask.shape[2]
    padw = math.ceil(mask.shape[3] / 8) * 8 - mask.shape[3]
    mask = F.pad(mask, [0, padw, 0, padh])
    return F.interpolate(F.max_pool2d(mask, 8, 8, 0),scale_factor=8,mode='nearest')[:,:,:-padh or None,:-padw or None]


class Model(nn.Module):
    def __init__(self, 
                 corr_radius=3, 
                 corr_lvls=4, 
                 num_flows=3, 
                 channels=[20, 32, 44, 56], 
                 skip_channels=20):
        super(Model, self).__init__()
        self.radius = corr_radius
        self.corr_levels = corr_lvls
        self.num_flows = num_flows
        self.channels = channels
        self.skip_channels = skip_channels

        self.feat_enc_outc = 84
        self.feat_encoder = SmallEncoder(output_dim=self.feat_enc_outc, norm_fn='instance', dropout=0.)
        self.encoder = Encoder(channels)

        self.decoder4 = InitDecoder(channels[3], channels[2], skip_channels)
        self.decoder3 = IntermediateDecoder(channels[2], channels[1], skip_channels)
        self.decoder2 = IntermediateDecoder(channels[1], channels[0], skip_channels)
        self.decoder1 = MultiFlowDecoder(channels[0], skip_channels, num_flows)

        self.update4 = self._get_updateblock(44)
        self.update3 = self._get_updateblock(32, 2)
        self.update2 = self._get_updateblock(20, 4)
        
        self.comb_block = nn.Sequential(
            nn.Conv2d(3*num_flows, 6*num_flows, 3, 1, 1),
            nn.PReLU(6*num_flows),
            nn.Conv2d(6*num_flows, 3, 3, 1, 1),
        )

        self.l1flow_weight = 4e-4
        self.thres_scale = 4

    def _get_updateblock(self, cdim, scale_factor=None):
        return SmallUpdateBlock(cdim=cdim, hidden_dim=76, flow_dim=20, corr_dim=64, 
                                fc_dim=68, scale_factor=scale_factor, 
                                corr_levels=self.corr_levels, radius=self.radius)


    def _corr_scale_lookup(self, corr_fn, coord, flow0, flow1, embt, downsample=1):
        # convert t -> 0 to 0 -> 1 | convert t -> 1 to 1 -> 0
        # based on linear assumption
        t1_scale = 1. / embt
        t0_scale = 1. / (1. - embt)
        if downsample != 1:
            inv = 1 / downsample
            flow0 = inv * resize(flow0, scale_factor=inv)
            flow1 = inv * resize(flow1, scale_factor=inv)
            
        corr0, corr1 = corr_fn(coord + flow1 * t1_scale, coord + flow0 * t0_scale) 
        corr = torch.cat([corr0, corr1], dim=1)
        flow = torch.cat([flow0, flow1], dim=1)
        return corr, flow


    def forward(self, img0, img1, embt, scale_factor=1.0, eval=False, **kwargs):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        img0_ = resize(img0, scale_factor) if scale_factor != 1.0 else img0
        img1_ = resize(img1, scale_factor) if scale_factor != 1.0 else img1
        b, _, h, w = img0_.shape
        coord = coords_grid(b, h // 8, w // 8, img0.device)
        
        fmap0, fmap1 = self.feat_encoder([img0_, img1_]) # [1, 128, H//8, W//8]
        corr_fn = BidirCorrBlock(fmap0, fmap1, radius=self.radius, num_levels=self.corr_levels)

        # f0_1: [1, c0, H//2, W//2] | f0_2: [1, c1, H//4, W//4]
        # f0_3: [1, c2, H//8, W//8] | f0_4: [1, c3, H//16, W//16]
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        ######################################### the 4th decoder #########################################
        up_flow0_4, up_flow1_4, ft_3_ = self.decoder4(f0_4, f1_4, embt)
        corr_4, flow_4 = self._corr_scale_lookup(corr_fn, coord, 
                                                 up_flow0_4, up_flow1_4, 
                                                 embt, downsample=1)

        # residue update with lookup corr
        delta_ft_3_, delta_flow_4 = self.update4(ft_3_, flow_4, corr_4)
        delta_flow0_4, delta_flow1_4 = torch.chunk(delta_flow_4, 2, 1)
        up_flow0_4 = up_flow0_4 + delta_flow0_4
        up_flow1_4 = up_flow1_4 + delta_flow1_4
        ft_3_ = ft_3_ + delta_ft_3_

        ######################################### the 3rd decoder #########################################
        up_flow0_3, up_flow1_3, ft_2_ = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        corr_3, flow_3 = self._corr_scale_lookup(corr_fn, 
                                                 coord, up_flow0_3, up_flow1_3, 
                                                 embt, downsample=2)

        # residue update with lookup corr
        delta_ft_2_, delta_flow_3 = self.update3(ft_2_, flow_3, corr_3)
        delta_flow0_3, delta_flow1_3 = torch.chunk(delta_flow_3, 2, 1)
        up_flow0_3 = up_flow0_3 + delta_flow0_3
        up_flow1_3 = up_flow1_3 + delta_flow1_3
        ft_2_ = ft_2_ + delta_ft_2_

        ######################################### the 2nd decoder #########################################
        up_flow0_2, up_flow1_2, ft_1_  = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        corr_2, flow_2 = self._corr_scale_lookup(corr_fn, 
                                                 coord, up_flow0_2, up_flow1_2, 
                                                 embt, downsample=4)
        
        # residue update with lookup corr
        delta_ft_1_, delta_flow_2 = self.update2(ft_1_, flow_2, corr_2)
        delta_flow0_2, delta_flow1_2 = torch.chunk(delta_flow_2, 2, 1)
        up_flow0_2 = up_flow0_2 + delta_flow0_2
        up_flow1_2 = up_flow1_2 + delta_flow1_2
        ft_1_ = ft_1_ + delta_ft_1_

        ######################################### the 1st decoder #########################################
        up_flow0_1, up_flow1_1, mask, img_res = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        
        if scale_factor != 1.0: 
            up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
            up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0/scale_factor)) * (1.0/scale_factor)
            mask = resize(mask, scale_factor=(1.0/scale_factor))
            img_res = resize(img_res, scale_factor=(1.0/scale_factor))
        
        # Merge multiple predictions 
        imgt_pred = multi_flow_combine(self.comb_block, img0, img1, up_flow0_1, up_flow1_1, 
                                                                        mask, img_res, mean_)
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        if eval:
            return  { 'imgt_pred': imgt_pred, }
        else:
            up_flow0_1 = up_flow0_1.reshape(b, self.num_flows, 2, h, w)
            up_flow1_1 = up_flow1_1.reshape(b, self.num_flows, 2, h, w)
            return {
                'imgt_pred': imgt_pred,
                'flow0_pred': [up_flow0_1, up_flow0_2, up_flow0_3, up_flow0_4],
                'flow1_pred': [up_flow1_1, up_flow1_2, up_flow1_3, up_flow1_4],
                'ft_pred': [ft_1_, ft_2_, ft_3_],
            }


    def forwardS(self, img0, img1, embt, scale_factor=1.0, eval=False, **kwargs):
        dfs = []
        dfs_mid = []
        loss_l1flow = 0

        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        img0_ = resize(img0, scale_factor) if scale_factor != 1.0 else img0
        img1_ = resize(img1, scale_factor) if scale_factor != 1.0 else img1
        b, _, h, w = img0_.shape
        coord = coords_grid(b, h // 8, w // 8, img0.device)

        fmap0, fmap1 = self.feat_encoder([img0_, img1_])  # [1, 128, H//8, W//8]
        corr_fn = BidirCorrBlock(fmap0, fmap1, radius=self.radius, num_levels=self.corr_levels)

        # f0_1: [1, c0, H//2, W//2] | f0_2: [1, c1, H//4, W//4]
        # f0_3: [1, c2, H//8, W//8] | f0_4: [1, c3, H//16, W//16]
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        ######################################### the 4th decoder #########################################
        up_flow0_4, up_flow1_4, ft_3_ = self.decoder4(f0_4, f1_4, embt)
        loss_l1flow += F.l1_loss(torch.cat([up_flow0_4, up_flow1_4], dim=1),
                                 torch.zeros(torch.cat([up_flow0_4, up_flow1_4], dim=1).shape,
                                             device='cuda')) * self.l1flow_weight
        corr_4, flow_4 = self._corr_scale_lookup(corr_fn, coord,
                                                 up_flow0_4, up_flow1_4,
                                                 embt, downsample=1)

        # residue update with lookup corr
        delta_ft_3_, delta_flow_4 = self.update4(ft_3_, flow_4, corr_4)
        delta_flow0_4, delta_flow1_4 = torch.chunk(delta_flow_4, 2, 1)
        up_flow0_4 = up_flow0_4 + delta_flow0_4
        up_flow1_4 = up_flow1_4 + delta_flow1_4
        ft_3_ = ft_3_ + delta_ft_3_
        up_mask_4 = ft_3_[:, :1]

        loss_l1flow += F.l1_loss(torch.cat([up_flow0_4,up_flow1_4],dim=1), torch.zeros(torch.cat([up_flow0_4,up_flow1_4],dim=1).shape, device='cuda')) * self.l1flow_weight
        dfs.append((up_flow0_4[:, 0:2]*8, up_flow1_4[:, 0:2]*8, ft_3_))


        ######################################### the 3rd decoder #########################################
        up_flow0_3, up_flow1_3, ft_2_, dflow0, dflow1 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        loss_l1flow += F.l1_loss(torch.cat([dflow0, dflow1], dim=1),
                                 torch.zeros(torch.cat([up_flow0_3, up_flow1_3], dim=1).shape,
                                             device='cuda')) * self.l1flow_weight
        corr_3, flow_3 = self._corr_scale_lookup(corr_fn,
                                                 coord, up_flow0_3, up_flow1_3,
                                                 embt, downsample=2)
        dfs_mid.append((dflow0[:, 0:2]*4, dflow1[:, 0:2]*4, ft_2_))

        # residue update with lookup corr
        delta_ft_2_, delta_flow_3 = self.update3(ft_2_, flow_3, corr_3)
        delta_flow0_3, delta_flow1_3 = torch.chunk(delta_flow_3, 2, 1)
        up_flow0_3 = up_flow0_3 + delta_flow0_3
        up_flow1_3 = up_flow1_3 + delta_flow1_3
        ft_2_ = ft_2_ + delta_ft_2_
        up_mask_3 = ft_2_[:, :1] + F.interpolate(up_mask_4, scale_factor=2, mode='nearest')
        loss_l1flow += F.l1_loss(torch.cat([dflow0+delta_flow0_3,dflow1+delta_flow1_3],dim=1), torch.zeros(torch.cat([up_flow0_3,up_flow1_3],dim=1).shape, device='cuda')) * self.l1flow_weight
        dfs.append(((delta_flow0_3+dflow0)[:, 0:2]*4, (delta_flow1_3+dflow1)[:, 0:2]*4, ft_2_))

        ######################################### the 2nd decoder #########################################
        up_flow0_2, up_flow1_2, ft_1_, dflow0, dflow1 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        loss_l1flow += F.l1_loss(torch.cat([dflow0, dflow1], dim=1),
                                 torch.zeros(torch.cat([up_flow0_2, up_flow1_2], dim=1).shape,
                                             device='cuda')) * self.l1flow_weight
        corr_2, flow_2 = self._corr_scale_lookup(corr_fn,
                                                 coord, up_flow0_2, up_flow1_2,
                                                 embt, downsample=4)
        dfs_mid.append((dflow0[:, 0:2]*2, dflow1[:, 0:2]*2, ft_1_))

        # residue update with lookup corr
        delta_ft_1_, delta_flow_2 = self.update2(ft_1_, flow_2, corr_2)
        delta_flow0_2, delta_flow1_2 = torch.chunk(delta_flow_2, 2, 1)
        up_flow0_2 = up_flow0_2 + delta_flow0_2
        up_flow1_2 = up_flow1_2 + delta_flow1_2
        ft_1_ = ft_1_ + delta_ft_1_
        up_mask_2 = ft_1_[:, :1] + F.interpolate(up_mask_3, scale_factor=2, mode='nearest')

        loss_l1flow += F.l1_loss(torch.cat([dflow0+delta_flow0_2,dflow1+delta_flow1_2],dim=1), torch.zeros(torch.cat([up_flow0_2,up_flow1_2],dim=1).shape, device='cuda')) * self.l1flow_weight
        dfs.append(((delta_flow0_2+dflow0)[:, 0:2]*2, (delta_flow1_2+dflow1)[:, 0:2]*2, ft_1_))

        ######################################### the 1st decoder #########################################
        up_flow0_1, up_flow1_1, mask, img_res, dflow0, dflow1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        mask = mask + F.interpolate(up_mask_2, scale_factor=2, mode='nearest')
        mask = torch.sigmoid(mask)
        loss_l1flow += F.l1_loss(torch.cat([dflow0,dflow1],dim=1), torch.zeros(torch.cat([up_flow0_1,up_flow1_1],dim=1).shape, device='cuda')) * self.l1flow_weight
        dfs.append((dflow0[:, ], dflow1[:, ], torch.cat([mask,img_res],dim=1)))


        if scale_factor != 1.0:
            up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
            up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
            mask = resize(mask, scale_factor=(1.0 / scale_factor))
            img_res = resize(img_res, scale_factor=(1.0 / scale_factor))

        # Merge multiple predictions
        imgt_pred = multi_flow_combine(self.comb_block, img0, img1, up_flow0_1, up_flow1_1,
                                       mask, img_res, mean_)
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        loss_dft = 0
        loss_sflow = 0
        loss_rflow = 0
        rflow_weight = 7e-5
        dft_weight = 1
        sflow_weight = 0.007

        # loss rflow
        for i in range(len(dfs)-2):
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

        for i in range(len(dfs) - 2):
            scale_dif1 = torch.max(torch.zeros(dfs[i + 1][0].shape, device='cuda'),
                      torch.abs(dfs[i + 1][0]) - F.interpolate(torch.abs(dfs[i][0]),
                                                               scale_factor=2))
            scale_dif2 = torch.max(torch.zeros(dfs[i + 1][1].shape, device='cuda'),
                      torch.abs(dfs[i + 1][1]) - F.interpolate(torch.abs(dfs[i][1]),
                                                               scale_factor=2))
            loss_sflow += sflow_weight * F.l1_loss(scale_dif2,torch.zeros(scale_dif2.shape, device='cuda'))
            loss_sflow += sflow_weight * F.l1_loss(scale_dif1,
                      torch.zeros(scale_dif1.shape, device='cuda'))

        for i in range(self.num_flows):
            scale_dif1 = torch.max(torch.zeros(dfs[3][0][:,2*i:2*i+2].shape, device='cuda'),
                      torch.abs(dfs[3][0][:,2*i:2*i+2]) - F.interpolate(torch.abs(dfs[2][0]),
                                                               scale_factor=2))
            scale_dif2 = torch.max(torch.zeros(dfs[3][1][:,2*i:2*i+2].shape, device='cuda'),
                      torch.abs(dfs[3][1][:,2*i:2*i+2]) - F.interpolate(torch.abs(dfs[2][1]),
                                                               scale_factor=2))
            loss_sflow += sflow_weight * F.l1_loss(scale_dif2,torch.zeros(scale_dif2.shape, device='cuda'))
            loss_sflow += sflow_weight * F.l1_loss(scale_dif1,
                      torch.zeros(scale_dif1.shape, device='cuda'))


        for i in range(len(dfs) - 2):
            scale_dif1 = torch.max(torch.zeros(dfs_mid[i][0].shape, device='cuda'),
                      torch.abs(dfs_mid[i][0]) - F.interpolate(torch.abs(dfs[i][0]),
                                                               scale_factor=2))

            scale_dif2 = torch.max(torch.zeros(dfs_mid[i][1].shape, device='cuda'),
                      torch.abs(dfs_mid[i][1]) - F.interpolate(torch.abs(dfs[i][1]),
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

        for i in range(len(dfs_mid)):
            flow_norm = torch.norm(torch.cat([dfs_mid[i][0],dfs_mid[i][1]],dim=1),p=1,dim=1)
            flow_norm = F.normalize(flow_norm, dim=(1, 2), p=1)
            ft_norm = torch.norm(dfs_mid[i][2],p=1,dim=1)
            ft_norm = F.normalize(ft_norm, dim=(1, 2), p=1)
            loss_dft += dft_weight * F.l1_loss(flow_norm, ft_norm)


        if eval:
            return {'imgt_pred': imgt_pred, }
        else:
            up_flow0_1 = up_flow0_1.reshape(b, self.num_flows, 2, h, w)
            up_flow1_1 = up_flow1_1.reshape(b, self.num_flows, 2, h, w)
            return {
                'imgt_pred': imgt_pred,
                'flow0_pred': [up_flow0_1, up_flow0_2, up_flow0_3, up_flow0_4],
                'flow1_pred': [up_flow1_1, up_flow1_2, up_flow1_3, up_flow1_4],
                'ft_pred': [ft_1_, ft_2_, ft_3_],
                'loss_dft' : loss_dft,
                'loss_sflow' : loss_sflow,
                'loss_rflow' : loss_rflow,
                'loss_l1flow' : loss_l1flow
            }


    def forwardS2(self, img0, img1, embt, scale_factor=1.0, eval=False, **kwargs):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        img0_ = resize(img0, scale_factor) if scale_factor != 1.0 else img0
        img1_ = resize(img1, scale_factor) if scale_factor != 1.0 else img1
        b, _, h, w = img0_.shape
        coord = coords_grid(b, h // 8, w // 8, img0.device)



        fmap0, fmap1 = self.feat_encoder([img0_, img1_])  # [1, 128, H//8, W//8]
        corr_fn = BidirCorrBlock(fmap0, fmap1, radius=self.radius, num_levels=self.corr_levels)

        # f0_1: [1, c0, H//2, W//2] | f0_2: [1, c1, H//4, W//4]
        # f0_3: [1, c2, H//8, W//8] | f0_4: [1, c3, H//16, W//16]
        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)

        loss_l1flow = 0

        ######################################### the 4th decoder #########################################
        up_flow0_4, up_flow1_4, ft_3_ = self.decoder4(f0_4, f1_4, embt)
        loss_l1flow += F.l1_loss(torch.cat([up_flow0_4, up_flow1_4], dim=1),
                                 torch.zeros(torch.cat([up_flow0_4, up_flow1_4], dim=1).shape,
                                             device='cuda')) * self.l1flow_weight
        corr_4, flow_4 = self._corr_scale_lookup(corr_fn, coord,
                                                 up_flow0_4, up_flow1_4,
                                                 embt, downsample=1)

        # residue update with lookup corr
        delta_ft_3_, delta_flow_4 = self.update4(ft_3_, flow_4, corr_4)
        delta_flow0_4, delta_flow1_4 = torch.chunk(delta_flow_4, 2, 1)
        up_flow0_4 = up_flow0_4 + delta_flow0_4
        up_flow1_4 = up_flow1_4 + delta_flow1_4
        ft_3_ = ft_3_ + delta_ft_3_
        up_mask_4 = ft_3_[:, :1]

        conv_thres = F.adaptive_max_pool2d(torch.abs(torch.cat([up_flow0_4,up_flow1_4],dim=1)),(1,1))/self.thres_scale
        conv_mask = ((torch.abs(up_flow0_4[:, 0:1]) < conv_thres[:,0:1] / 8) & (
                torch.abs(up_flow0_4[:, 1:2]) < conv_thres[:,1:2] / 8) & (torch.abs(up_flow1_4[:, 0:1]) < conv_thres[:,2:3] / 8)
                     & (torch.abs(up_flow1_4[:, 1:2]) < conv_thres[:,3:4] / 8)).float()        
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2)

        #conv_mask_noise = (F.avg_pool2d(conv_mask,kernel_size=11,stride=1,padding=5)>0.2).float()
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=15, stride=1, padding=7) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        loss_l1flow += F.l1_loss(torch.cat([up_flow0_4, up_flow1_4],dim=1), torch.zeros(torch.cat([up_flow0_4, up_flow1_4],dim=1).shape, device='cuda')) * self.l1flow_weight


        ######################################### the 3rd decoder #########################################
        up_flow0_3, up_flow1_3, ft_2_, dflow0, dflow1, odflow0, odflow1 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4, conv_mask)
        loss_l1flow += F.l1_loss(torch.cat([odflow0, odflow1], dim=1),
                                 torch.zeros(torch.cat([up_flow0_3, up_flow1_3], dim=1).shape,
                                             device='cuda')) * self.l1flow_weight
        corr_3, flow_3 = self._corr_scale_lookup(corr_fn,
                                                 coord, up_flow0_3, up_flow1_3,
                                                 embt, downsample=2)

        # residue update with lookup corr
        delta_ft_2_, delta_flow_3 = self.update3(ft_2_, flow_3, corr_3)
        delta_flow0_3, delta_flow1_3 = torch.chunk(delta_flow_3, 2, 1)
        up_flow0_3 = up_flow0_3 + delta_flow0_3
        up_flow1_3 = up_flow1_3 + delta_flow1_3
        ft_2_ = ft_2_ + delta_ft_2_
        up_mask_3 = ft_2_[:, :1] + F.interpolate(up_mask_4, scale_factor=2, mode='nearest')

        
        conv_thres = F.adaptive_max_pool2d(torch.abs(torch.cat([dflow0+delta_flow0_3,dflow1+delta_flow1_3],dim=1)),(1,1))/self.thres_scale
        conv_mask = ((torch.abs((dflow0+delta_flow0_3)[:, 0:1]* conv_mask) < conv_thres[:,0:1] / 4) & (
                torch.abs((dflow0+delta_flow0_3))[:, 1:2]* conv_mask < conv_thres[:,1:2] / 4) & (torch.abs((dflow1+delta_flow1_3)[:, 0:1]* conv_mask) < conv_thres[:,2:3] / 4)
                     & (torch.abs((dflow1+delta_flow1_3)[:, 1:2]* conv_mask) < conv_thres[:,3:4] / 4)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2)
        # conv_mask_noise = (F.avg_pool2d(conv_mask,kernel_size=11,stride=1,padding=5)>0.2).float()
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=29, stride=1, padding=14) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        loss_l1flow += F.l1_loss(torch.cat([(delta_flow0_3+odflow0), (delta_flow1_3+odflow1)],dim=1), torch.zeros(torch.cat([up_flow0_3, up_flow1_3],dim=1).shape, device='cuda')) * self.l1flow_weight

        ######################################### the 2nd decoder #########################################
        up_flow0_2, up_flow1_2, ft_1_, dflow0, dflow1, odflow0, odflow1 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3, conv_mask)
        loss_l1flow += F.l1_loss(torch.cat([odflow0, odflow1], dim=1),
                                 torch.zeros(torch.cat([up_flow0_2, up_flow1_2], dim=1).shape,
                                             device='cuda')) * self.l1flow_weight
        corr_2, flow_2 = self._corr_scale_lookup(corr_fn,
                                                 coord, up_flow0_2, up_flow1_2,
                                                 embt, downsample=4)

        # residue update with lookup corr
        delta_ft_1_, delta_flow_2 = self.update2(ft_1_, flow_2, corr_2)
        delta_flow0_2, delta_flow1_2 = torch.chunk(delta_flow_2, 2, 1)
        up_flow0_2 = up_flow0_2 + delta_flow0_2
        up_flow1_2 = up_flow1_2 + delta_flow1_2
        ft_1_ = ft_1_ + delta_ft_1_
        up_mask_2 = ft_1_[:, :1] + F.interpolate(up_mask_3, scale_factor=2, mode='nearest')
        
        conv_thres = F.adaptive_max_pool2d(torch.abs(torch.cat([dflow0+delta_flow0_2,dflow1+delta_flow1_2],dim=1)),(1,1))/self.thres_scale*0.5
        conv_mask = ((torch.abs((dflow0+delta_flow0_2)[:, 0:1]) < conv_thres[:,0:1] / 2) & (
                torch.abs((dflow0+delta_flow0_2)[:, 1:2]) < conv_thres[:,1:2] / 2) & (torch.abs((dflow1+delta_flow1_2)[:, 0:1]) < conv_thres[:,2:3] / 2)
                     & (torch.abs((dflow0+delta_flow1_2)[:, 1:2]) < conv_thres[:,3:4] / 2)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2)
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=57, stride=1, padding=28) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        loss_l1flow += F.l1_loss(torch.cat([(delta_flow0_2+odflow0), (delta_flow1_2+odflow1)],dim=1), torch.zeros(torch.cat([up_flow0_2, up_flow1_2],dim=1).shape, device='cuda')) * self.l1flow_weight


        ######################################### the 1st decoder #########################################
        up_flow0_1, up_flow1_1, mask, img_res, dflow0, dflow1, odflow0, odflow1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2, conv_mask)
        loss_l1flow += F.l1_loss(torch.cat([odflow0, odflow1], dim=1),
                                 torch.zeros(torch.cat([up_flow0_1, up_flow1_1], dim=1).shape,
                                             device='cuda')) * self.l1flow_weight
        mask = mask + F.interpolate(up_mask_2, scale_factor=2, mode='nearest')
        mask = torch.sigmoid(mask)
        if scale_factor != 1.0:
            up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
            up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
            mask = resize(mask, scale_factor=(1.0 / scale_factor))
            img_res = resize(img_res, scale_factor=(1.0 / scale_factor))

        # Merge multiple predictions
        imgt_pred = multi_flow_combine(self.comb_block, img0, img1, up_flow0_1, up_flow1_1,
                                       mask, img_res, mean_)
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        if eval:
            return {'imgt_pred': imgt_pred, }
        else:
            up_flow0_1 = up_flow0_1.reshape(b, self.num_flows, 2, h, w)
            up_flow1_1 = up_flow1_1.reshape(b, self.num_flows, 2, h, w)
            return {
                'imgt_pred': imgt_pred,
                'flow0_pred': [up_flow0_1, up_flow0_2, up_flow0_3, up_flow0_4],
                'flow1_pred': [up_flow1_1, up_flow1_2, up_flow1_3, up_flow1_4],
                'ft_pred': [ft_1_, ft_2_, ft_3_],
                'lossl1_flow': loss_l1flow
            }


    def get_dynamic_macs(self, img0, img1, embt, scale_factor=1.0, thres=15):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        img0_ = resize(img0, scale_factor) if scale_factor != 1.0 else img0
        img1_ = resize(img1, scale_factor) if scale_factor != 1.0 else img1
        b, _, h, w = img0_.shape
        coord = coords_grid(b, h // 8, w // 8, img0.device)

        fmap0, fmap1 = self.feat_encoder([img0_, img1_])  # [1, 128, H//8, W//8]
        corr_fn = BidirCorrBlock(fmap0, fmap1, radius=self.radius, num_levels=self.corr_levels)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_)
        H, W = img0_.shape[2], img0_.shape[3]
        conv_masks = []

        ######################################### the 4th decoder #########################################
        up_flow0_4, up_flow1_4, ft_3_ = self.decoder4(f0_4, f1_4, embt)
        corr_4, flow_4 = self._corr_scale_lookup(corr_fn, coord,
                                                 up_flow0_4, up_flow1_4,
                                                 embt, downsample=1)

        # residue update with lookup corr
        delta_ft_3_, delta_flow_4 = self.update4(ft_3_, flow_4, corr_4)
        delta_flow0_4, delta_flow1_4 = torch.chunk(delta_flow_4, 2, 1)
        up_flow0_4 = up_flow0_4 + delta_flow0_4
        up_flow1_4 = up_flow1_4 + delta_flow1_4
        ft_3_ = ft_3_ + delta_ft_3_
        up_mask_4 = ft_3_[:, :1]

        conv_thres = F.adaptive_max_pool2d(torch.abs(torch.cat([up_flow0_4, up_flow1_4], dim=1)),
                                           (1, 1)) / thres
        conv_mask = ((torch.abs(up_flow0_4[:, 0:1]) < conv_thres[:, 0:1] / 8) & (
                torch.abs(up_flow0_4[:, 1:2]) < conv_thres[:, 1:2] / 8) & (
                                 torch.abs(up_flow1_4[:, 0:1]) < conv_thres[:, 2:3] / 8)
                     & (torch.abs(up_flow1_4[:, 1:2]) < conv_thres[:, 3:4] / 8)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2)

        # conv_mask_noise = (F.avg_pool2d(conv_mask,kernel_size=11,stride=1,padding=5)>0.2).float()
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=15, stride=1, padding=7) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        conv_masks.append(conv_mask)

        ######################################### the 3rd decoder #########################################
        up_flow0_3, up_flow1_3, ft_2_, dflow0, dflow1, odflow0, odflow1 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4,
                                                                                        up_flow1_4, conv_mask)
        corr_3, flow_3 = self._corr_scale_lookup(corr_fn,
                                                 coord, up_flow0_3, up_flow1_3,
                                                 embt, downsample=2)

        # residue update with lookup corr
        delta_ft_2_, delta_flow_3 = self.update3(ft_2_, flow_3, corr_3)
        delta_flow0_3, delta_flow1_3 = torch.chunk(delta_flow_3, 2, 1)
        up_flow0_3 = up_flow0_3 + delta_flow0_3
        up_flow1_3 = up_flow1_3 + delta_flow1_3
        ft_2_ = ft_2_ + delta_ft_2_
        up_mask_3 = ft_2_[:, :1] + F.interpolate(up_mask_4, scale_factor=2, mode='nearest')

        conv_thres = F.adaptive_max_pool2d(
            torch.abs(torch.cat([dflow0 + delta_flow0_3, dflow1 + delta_flow1_3], dim=1)), (1, 1)) / thres
        conv_mask = ((torch.abs((dflow0 + delta_flow0_3)[:, 0:1] * conv_mask) < conv_thres[:, 0:1] / 4) & (
                torch.abs((dflow0 + delta_flow0_3))[:, 1:2] * conv_mask < conv_thres[:, 1:2] / 4) & (
                                 torch.abs((dflow1 + delta_flow1_3)[:, 0:1] * conv_mask) < conv_thres[:, 2:3] / 4)
                     & (torch.abs((dflow1 + delta_flow1_3)[:, 1:2] * conv_mask) < conv_thres[:, 3:4] / 4)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2)
        # conv_mask_noise = (F.avg_pool2d(conv_mask,kernel_size=11,stride=1,padding=5)>0.2).float()
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=29, stride=1, padding=14) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        conv_masks.append(conv_mask)

        ######################################### the 2nd decoder #########################################
        up_flow0_2, up_flow1_2, ft_1_, dflow0, dflow1, odflow0, odflow1 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3,
                                                                                        up_flow1_3, conv_mask)
        corr_2, flow_2 = self._corr_scale_lookup(corr_fn,
                                                 coord, up_flow0_2, up_flow1_2,
                                                 embt, downsample=4)

        # residue update with lookup corr
        delta_ft_1_, delta_flow_2 = self.update2(ft_1_, flow_2, corr_2)
        delta_flow0_2, delta_flow1_2 = torch.chunk(delta_flow_2, 2, 1)
        up_flow0_2 = up_flow0_2 + delta_flow0_2
        up_flow1_2 = up_flow1_2 + delta_flow1_2
        ft_1_ = ft_1_ + delta_ft_1_
        up_mask_2 = ft_1_[:, :1] + F.interpolate(up_mask_3, scale_factor=2, mode='nearest')

        conv_thres = F.adaptive_max_pool2d(
            torch.abs(torch.cat([dflow0 + delta_flow0_2, dflow1 + delta_flow1_2], dim=1)),
            (1, 1)) / thres * 0.5
        conv_mask = ((torch.abs((dflow0 + delta_flow0_2)[:, 0:1]) < conv_thres[:, 0:1] / 2) & (
                torch.abs((dflow0 + delta_flow0_2)[:, 1:2]) < conv_thres[:, 1:2] / 2) & (
                                 torch.abs((dflow1 + delta_flow1_2)[:, 0:1]) < conv_thres[:, 2:3] / 2)
                     & (torch.abs((dflow0 + delta_flow1_2)[:, 1:2]) < conv_thres[:, 3:4] / 2)).float()
        conv_mask = 1 - F.interpolate(conv_mask, scale_factor=2)
        conv_mask_noise = (F.avg_pool2d(conv_mask, kernel_size=57, stride=1, padding=28) > 0.3).float()
        conv_mask = conv_mask_noise * conv_mask
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        conv_masks.append(conv_mask)

        ######################################### the 1st decoder #########################################
        up_flow0_1, up_flow1_1, mask, img_res, dflow0, dflow1, odflow0, odflow1 = self.decoder1(ft_1_, f0_1, f1_1,
                                                                                                up_flow0_2, up_flow1_2,
                                                                                                conv_mask)
        mask = mask + F.interpolate(up_mask_2, scale_factor=2, mode='nearest')
        mask = torch.sigmoid(mask)
        if scale_factor != 1.0:
            up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
            up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
            mask = resize(mask, scale_factor=(1.0 / scale_factor))
            img_res = resize(img_res, scale_factor=(1.0 / scale_factor))

        # Merge multiple predictions
        imgt_pred = multi_flow_combine(self.comb_block, img0, img1, up_flow0_1, up_flow1_1,
                                       mask, img_res, mean_)
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        #CorrBlock
        F_corr = self.feat_enc_outc*H*W*H*W/8/8/8/8 * 1e-9

        # Encoder FLOPs
        orH, orW =H, W
        F_encoder = 0
        for n, m in self.encoder.named_modules():
            if type(m) is nn.Conv2d:
                if m.stride[0] == 2:
                    H /= 2
                    W /= 2
                F_encoder += H * W * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9
            elif type(m) is nn.PReLU:
                c_in = m.num_parameters
                F_encoder += c_in * H * W * 1e-9
        F_encoder *= 2  # two images

        #feat Encoder
        F_featEncoder = 0
        for n, m in self.feat_encoder.named_modules():
            if type(m) is nn.Conv2d:
                if m.stride[0] == 2:
                    orH /= 2
                    orW /= 2
                F_encoder += orH * orW * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9
            elif type(m) is nn.PReLU:
                c_in = m.num_parameters
                F_encoder += c_in * orH * orW * 1e-9
            elif type(m) is nn.InstanceNorm2d:
                c_in = m.num_features
                F_encoder += orH * orW * c_in * 1e-9

        # Decoder4 FLOPs
        F_decoder4 = 0
        for n, m in self.decoder4.named_modules():
            if type(m) is nn.Conv2d:
                F_decoder4 += H * W * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9
            elif type(m) is nn.PReLU:
                c_in = m.num_parameters
                F_decoder4 += c_in * H * W * 1e-9
            elif type(m) is nn.ConvTranspose2d:  # stride==2
                H *= 2
                W *= 2
                F_decoder4 += H * W * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9

        for n, m in self.update4.named_modules():
            if type(m) is nn.Conv2d:
                F_decoder4 += H * W * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9


        # Decoder3 FLOPs
        conv_mask = conv_masks[0]
        conv_mask_list = []
        conv_mask_list.append(get_forward_mask(conv_mask))
        conv_mask = F.max_pool2d(conv_mask, 4, 2, 1)
        conv_mask_list.append(get_forward_mask(conv_mask))
        for i in range(5):
            conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
            conv_mask_list.append(get_forward_mask(conv_mask))
        conv_mask_index = 6
        F_decoder3 = 0
        for n, m in self.decoder3.named_modules():
            if type(m) is nn.Conv2d:
                F_decoder3 += (torch.sum(conv_mask_list[
                                             conv_mask_index])  # (torch.sum(conv_mask_list[conv_mask_index]) * 8 * 8 = H * W?
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
                conv_mask_index -= 1
            elif type(m) is nn.PReLU:
                c_in = m.num_parameters
                F_decoder3 += c_in * H * W * 1e-9
            elif type(m) is nn.ConvTranspose2d:  # stride==2
                H *= 2
                W *= 2
                F_decoder3 += (torch.sum(conv_mask_list[conv_mask_index])
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)

        F_decoder3 += self.get_update_macs(self.update3, conv_masks[0], H, W)


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
                F_decoder2 += (torch.sum(conv_mask_list[
                                             conv_mask_index])  # (torch.sum(conv_mask_list[conv_mask_index]) * 8 * 8 = H * W?
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
                conv_mask_index -= 1
            elif type(m) is nn.PReLU:
                c_in = m.num_parameters
                F_decoder2 += c_in * H * W * 1e-9
            elif type(m) is nn.ConvTranspose2d:  # stride==2
                H *= 2
                W *= 2
                F_decoder2 += (torch.sum(conv_mask_list[conv_mask_index])
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
        F_decoder2 += self.get_update_macs(self.update2, conv_masks[1], H, W)

        # Decoder1 FLOPs
        conv_mask = conv_masks[2]
        conv_mask_list = []
        conv_mask_list.append(get_forward_mask(conv_mask))
        conv_mask = F.max_pool2d(conv_mask, 4, 2, 1)
        conv_mask_list.append(get_forward_mask(conv_mask))
        for i in range(5):
            conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
            conv_mask_list.append(get_forward_mask(conv_mask))
        conv_mask_index = 6
        F_decoder1 = 0
        for n, m in self.decoder1.named_modules():
            if type(m) is nn.Conv2d:
                F_decoder1 += (torch.sum(conv_mask_list[
                                             conv_mask_index])  # (torch.sum(conv_mask_list[conv_mask_index]) * 8 * 8 = H * W?
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
                conv_mask_index -= 1
            elif type(m) is nn.PReLU:
                c_in = m.num_parameters
                F_decoder1 += c_in * H * W * 1e-9
            elif type(m) is nn.ConvTranspose2d:  # stride==2
                H *= 2
                W *= 2
                F_decoder1 += (torch.sum(conv_mask_list[conv_mask_index])
                               * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)

        F_model = F_encoder + F_decoder4 + F_decoder3 + F_decoder2 + F_decoder1 + F_corr
        return imgt_pred, F_model


    def get_update_macs(self, block, conv_mask, H, W):
        F_decoder = 0
        conv_mask_list = []
        #flow feat head-7,6
        conv_mask_list.append(get_forward_mask(conv_mask))
        m = block.flow_head[2]
        F_decoder += (torch.sum(conv_mask_list[
                                     -1])
                       * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
        m = block.feat_head[2]
        F_decoder += (torch.sum(conv_mask_list[
                                    -1])
                      * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
        conv_mask=F.max_pool2d(conv_mask, 3, 1, 1)
        conv_mask_list.append(get_forward_mask(conv_mask))
        m = block.flow_head[0]
        F_decoder += (torch.sum(conv_mask_list[
                                     -1])
                       * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
        m = block.feat_head[0]
        F_decoder += (torch.sum(conv_mask_list[
                                     -1])
                       * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)

        #gru-5,4
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        conv_mask_list.append(get_forward_mask(conv_mask))
        m = block.gru[2]
        F_decoder += (torch.sum(conv_mask_list[
                                     -1])
                       * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        conv_mask_list.append(get_forward_mask(conv_mask))
        m = block.gru[0]
        F_decoder += (torch.sum(conv_mask_list[
                                     -1])
                       * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
        #conv-3
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        conv_mask_list.append(get_forward_mask(conv_mask))
        m = block.conv
        F_decoder += (torch.sum(conv_mask_list[
                                     -1])
                       * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
        temp = conv_mask.clone()
        #convf2-2
        conv_mask = F.max_pool2d(conv_mask, 3, 1, 1)
        conv_mask_list.append(get_forward_mask(conv_mask))
        m = block.convf2
        F_decoder += (torch.sum(conv_mask_list[
                                     -1])
                       * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
        #convf1-1
        conv_mask = F.max_pool2d(conv_mask, 7, 1, 3)
        conv_mask_list.append(get_forward_mask(conv_mask))
        m = block.convf1
        F_decoder += (torch.sum(conv_mask_list[
                                     -1])
                       * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
        #convc1-0
        conv_mask_list.append(get_forward_mask(conv_mask))
        m = block.convc1
        F_decoder += (torch.sum(conv_mask_list[
                                     -1])
                       * (m.in_channels * m.kernel_size[0] * m.kernel_size[1] + 1) * m.out_channels * 1e-9)
        return F_decoder
