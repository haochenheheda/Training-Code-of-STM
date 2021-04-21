from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
 
# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys

from resnest.torch import resnest101
from utils.helpers import *
 


class ResBlock(nn.Module):
    def __init__(self, backbone, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = backbone
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        if self.backbone == 'resnest101':
            r = self.conv1(F.relu(x,inplace=True))
            r = self.conv2(F.relu(r,inplace=True))
        else:
            r = self.conv1(F.relu(x))
            r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Encoder_M(nn.Module):
    def __init__(self,backbone):
        super(Encoder_M, self).__init__()
        if backbone == 'resnest101':
            self.conv1_m = nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1_o = nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif backbone == 'resnest101':
            resnet = resnest101()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float() # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f
 
class Encoder_Q(nn.Module):
    def __init__(self, backbone):
        super(Encoder_Q, self).__init__()

        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif backbone == 'resnest101':
            resnet = resnest101()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, backbone, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(backbone,planes, planes)
        self.ResMM = ResBlock(backbone,planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim,scale_rate,backbone):
        super(Decoder, self).__init__()
        self.backbone = backbone
        if backbone == 'resnest101':
            self.convFM = nn.Conv2d(256, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        else:
            self.convFM = nn.Conv2d(1024//scale_rate, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(backbone,mdim, mdim)
        self.RF3 = Refine(backbone, 512//scale_rate, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(backbone, 256//scale_rate, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)


    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        if self.backbone == 'resnest101':
            p2 = self.pred2(F.relu(m2,inplace=True))
        else:
            p2 = self.pred2(F.relu(m2))
        
        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p #, p2, p3, p4

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)

    def forward(self, x):
        x = self.atrous_conv(x)
        return F.relu(x,inplace=True)

class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        dilations = [1, 2, 4, 8]

        self.aspp1 = _ASPPModule(1024, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(1024, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(1024, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(1024, 256, 3, padding=dilations[3], dilation=dilations[3])
        self.conv1 = nn.Conv2d(1024, 256, 1, bias=False)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x)
        return F.dropout(F.relu(x,inplace = True),p = 0.5,training=self.training)   

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W) 
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb
 
        qi = q_in.view(B, D_e, H*W)  # b, emb, HW
 
        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1) # b, THW, HW

        mo = m_out.view(B, D_o, T*H*W) 
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p

class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)




class STM(nn.Module):
    def __init__(self,backbone = 'resnet50'):
        super(STM, self).__init__()
        self.backbone = backbone
        assert backbone == 'resnet50' or backbone == 'resnet18' or backbone == 'resnest101'
        scale_rate = (1 if (backbone == 'resnet50' or backbone == 'resnest101') else 4)

        self.Encoder_M = Encoder_M(backbone) 
        self.Encoder_Q = Encoder_Q(backbone) 

        self.KV_M_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)
        self.KV_Q_r4 = KeyValue(1024//scale_rate, keydim=128//scale_rate, valdim=512//scale_rate)

        self.Memory = Memory()
        self.Decoder = Decoder(256,scale_rate,backbone)
        if backbone == 'resnest101':
            self.aspp = ASPP()
 
    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            pad_mem = ToCuda(torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            pad_mem[0,1:num_objects+1,:,0] = mem
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, frame, masks, num_objects): 
        # memorize a frame 
        num_objects = num_objects[0].item()
        _, K, H, W = masks.shape # B = 1

        (frame, masks), pad = pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))

        # make batch arg list
        B_list = {'f':[], 'm':[], 'o':[]}
        for o in range(1, num_objects+1): # 1 - no
            B_list['f'].append(frame)
            B_list['m'].append(masks[:,o])
            B_list['o'].append( (torch.sum(masks[:,1:o], dim=1) + \
                torch.sum(masks[:,o+1:num_objects+1], dim=1)).clamp(0,1) )

        # make Batch
        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)

        r4, _, _, _, _ = self.Encoder_M(B_['f'], B_['m'], B_['o'])
        k4, v4 = self.KV_M_r4(r4) # num_objects, 128 and 512, H/16, W/16
        k4, v4 = self.Pad_memory([k4, v4], num_objects=num_objects, K=K)
        return k4, v4

    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = ToCuda(torch.zeros(1, K, H, W)) 
        em[0,0] =  torch.prod(1-ps, dim=0) # bg prob
        em[0,1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit

    def segment(self, frame, keys, values, num_objects): 
        num_objects = num_objects[0].item()
        _, K, keydim, T, H, W = keys.shape # B = 1
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        
        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
        r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)
        
        # memory select kv:(1, K, C, T, H, W)
        m4, viz = self.Memory(keys[0,1:num_objects+1], values[0,1:num_objects+1], k4e, v4e)
        if self.backbone == 'resnest101':
            m4 = self.aspp(m4)
        logits = self.Decoder(m4, r3e, r2e)
        ps = F.softmax(logits, dim=1)[:,1] # no, h, w  
        #ps = indipendant possibility to belong to each object
        
        logit = self.Soft_aggregation(ps, K) # 1, K, H, W

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]

        return logit    

    def forward(self, *args, **kwargs):
        if args[1].dim() > 4: # keys
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)