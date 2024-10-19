import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from einops import rearrange, repeat
import torch.nn.functional as F
import math
from mmcv.cnn import ConvModule





class SemiDecoder(nn.Module):
    """
    SegFormer V16: cross attn with memory bank storing unlabeled data
    """

    def __init__(self, num_class, in_planes, embedding_dim, **kwargs):
        super(SemiDecoder, self).__init__()

        self.decoder = SegFormerHead(embedding_dim, in_planes, num_class)



    def forward(self, feats, h, w,need_fp=False,  feature_scale=None):
        e1, e2, e3, e4 = feats
        att_feat, feature = self.decoder(e1, e2, e3, e4)

        if need_fp:
            if feature_scale == 1.0:
                e1_fp, e2_fp, e3_fp, e4_fp = e1, e2, e3, e4
            else:
                e1_fp = F.interpolate(e1, scale_factor=feature_scale, mode="bilinear", align_corners=True)
                e2_fp = F.interpolate(e2, scale_factor=feature_scale, mode="bilinear", align_corners=True)
                e3_fp = F.interpolate(e3, scale_factor=feature_scale, mode="bilinear", align_corners=True)
                e4_fp = F.interpolate(e4, scale_factor=feature_scale, mode="bilinear", align_corners=True)
            outs, features = self.decoder(torch.cat((e1_fp, nn.Dropout2d(0.5)(e1_fp))),
                                torch.cat((e2_fp, nn.Dropout2d(0.5)(e2_fp))),
                                torch.cat((e3_fp, nn.Dropout2d(0.5)(e3_fp))),
                                torch.cat((e4_fp, nn.Dropout2d(0.5)(e4_fp))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            feature, feature_fp = features.chunk(2)
            return out, feature, out_fp

        out = F.interpolate(att_feat, size=(h, w), mode="bilinear", align_corners=False)


        return out, feature

class DeepLabV3plusHead(nn.Module):
    """
    SegFormer V16: cross attn with memory bank storing unlabeled data
    """

    def __init__(self, mid_channels=256, high_channels=2048):
        super(DeepLabV3plusHead, self).__init__()


        self.reduce = nn.Sequential(nn.Conv2d(mid_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))


    def forward(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out, feature



class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, embedding_dim, in_channels, num_class):
        super(SegFormerHead, self).__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels


        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        
        self.linear_reduce = ConvModule(
            in_channels=embedding_dim,
            out_channels=256,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.dropout = nn.Dropout2d(0.1)

        self.linear_pred = nn.Conv2d(embedding_dim, num_class, kernel_size=1)

    def forward(self, c1, c2, c3, c4):
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).contiguous().reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).contiguous().reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).contiguous().reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        
        feature =  self.linear_reduce(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x, feature