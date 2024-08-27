import model.backbone.resnet as resnet
from model.backbone.xception import xception
# from einops.layers.torch import Rearrange
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.helpers import named_apply
from functools import partial
from copy import deepcopy
from einops import rearrange
import torch
from torch import nn
from mmengine import MODELS
from mmengine.model import BaseModule
from timm.layers import DropPath, get_norm_layer

class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        # if 'resnet' in cfg['backbone']:
        self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True,
                                                         replace_stride_with_dilation=cfg[
                                                             'replace_stride_with_dilation'])
        # else:
        #     assert cfg['backbone'] == 'xception'
        #     self.backbone = xception(pretrained=True)
        self.input_size = cfg['crop_size']
        low_channels = 128
        mid_channels = 256
        high_channels = 2048
        scale_in_ch = 2 * mid_channels

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(mid_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))


        self.scale_attn = nn.Sequential(
            nn.Conv2d(scale_in_ch + 32, scale_in_ch + 32, kernel_size=3, padding=1, groups=scale_in_ch + 32, bias=False),  # 深度卷积
            nn.BatchNorm2d(scale_in_ch + 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale_in_ch + 32, 128, kernel_size=1, bias=False),  # 逐点卷积
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False),  # 深度卷积
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, bias=False),  # 逐点卷积
            nn.Sigmoid())

        self.se_block = SqueezeExcitation(scale_in_ch + 32)
        self.RWKV_layers = RWKVB_layers(1, scale_in_ch // 16, mlp_ratio=4.0, drop_path=0., total_layers=2)
        

        self.classifier = nn.Conv2d(mid_channels, cfg['nclass'], 1, bias=True)

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out, feature

    def _fwd(self, x, need_fp=False, scale_factor=None, feature_scale=None):
        h, w = x.shape[-2:]
        feats = self.backbone.base_forward(x)

        c1, c4 = feats[0], feats[-1]

        if need_fp:
            if feature_scale == 1.0:
                c1_fp, c4_fp = c1, c4
            else:
                c1_fp = F.interpolate(c1, scale_factor=feature_scale, mode="bilinear", align_corners=True)
                c4_fp = F.interpolate(c4, scale_factor=feature_scale, mode="bilinear", align_corners=True)
            outs, features = self._decode(torch.cat((c1_fp, nn.Dropout2d(0.5)(c1_fp))), torch.cat((c4_fp, nn.Dropout2d(0.5)(c4_fp))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            feature, _ = features.chunk(2)
            return out, feature, out_fp

        out, feature = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out, feature

    def two_scale_forward(self, inputs, scale_factor, feature_scale):
        x_1x = inputs
        B, C, H, W = x_1x.shape
        if scale_factor == None:
            out, _ = self._fwd(x_1x)
            return out

        if scale_factor > 1.0:
            x_lo = x_1x
            x_hi = ResizeX(x_1x, scale_factor)
            p_lo_ori, feats_lo, out_fp = self._fwd(x_lo, need_fp=True, feature_scale=feature_scale)
            p_hi, feats_hi = self._fwd(x_hi)
            p_hi = scale_as(p_hi, x_1x)
            feats_hi = scale_as(feats_hi, feats_lo)
            cat_feats = torch.cat([feats_lo, feats_hi], 1) # channel 256

            H, W = cat_feats.size(2), cat_feats.size(3)
            
            global_int_feats = self.RWKV_layers(cat_feats, H, W)
            global_int_feats =  rearrange(global_int_feats, "b (h w) c -> b c h w", h=H, w=W).contiguous()

            channel_attn_feats = self.se_block(torch.cat([cat_feats, global_int_feats], 1))

            logit_attn = self.scale_attn(channel_attn_feats)
            logit_attn = scale_as(logit_attn, p_lo_ori)
            p_lo = logit_attn * p_lo_ori
            p_lo_up = scale_as(p_lo, p_hi)
            logit_attn = scale_as(logit_attn, p_hi)
            joint_pred = p_lo_up + (1 - logit_attn) * p_hi
            joint_pred = scale_as(joint_pred, p_lo)

            return {
                'pred_joint': joint_pred,
                'pred_ori': p_lo_ori,
                'pred_fp': out_fp,
                'pred_size': p_hi}

        else:
            x_lo = ResizeX(x_1x, scale_factor)
            x_hi = x_1x

            p_lo, feats_lo = self._fwd(x_lo, scale_factor=scale_factor)
            p_hi, feats_hi, out_fp = self._fwd(x_hi, need_fp=True, feature_scale=feature_scale)
            p_lo_ori = scale_as(p_lo, x_1x)
            feats_lo = scale_as(feats_lo, feats_hi)
            cat_feats = torch.cat([feats_lo, feats_hi], 1)
            H, W = cat_feats.size(2), cat_feats.size(3)
            
            global_int_feats = self.RWKV_layers(cat_feats, H, W)
            global_int_feats =  rearrange(global_int_feats, "b (h w) c -> b c h w", h=H, w=W).contiguous()
            channel_attn_feats = self.se_block(torch.cat([cat_feats, global_int_feats], 1))

            
            logit_attn = self.scale_attn(channel_attn_feats)
            logit_attn = scale_as(logit_attn, p_lo)
            p_lo_att = logit_attn * p_lo
            p_lo_att = scale_as(p_lo_att, p_hi)
            logit_attn = scale_as(logit_attn, p_hi)
            joint_pred = p_lo_att + (1 - logit_attn) * p_hi

            return {
                'pred_joint': joint_pred,
                'pred_ori': p_hi,
                'pred_fp': out_fp,
                'pred_size': p_lo_ori}

    def forward(self, inputs, scale_factor=None, feature_scale=1.0, scales=None, eval_mode='atten_fusion'):
        # cfg.MODEL.N_SCALES 多尺度列表
        # if scales and not self.training:
        if scales:
            return self.nscale_forward(inputs, scales, eval_mode)

        return self.two_scale_forward(inputs, scale_factor, feature_scale)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, num_channels, H, W = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        scale = x * excitation
        return scale


class MultiHeadCoAttention(nn.Module):
    def __init__(self, cfg, in_channels, d_model, num_heads):
        super(MultiHeadCoAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.mlp = nn.Sequential(nn.Linear(in_channels, d_model), nn.LayerNorm(d_model))
        # self.mlp2 = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.Q_norm = nn.LayerNorm(in_channels)
        self.K_norm = nn.LayerNorm(in_channels)
        self.V_norm = nn.LayerNorm(in_channels)
        self.scale_fusion = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, dropout=0.)

    def forward(self, x):
        batch_size = x.size(0)
        H, W = x.size(2), x.size(3)

        Q, K, V = x, x, x

        Q = rearrange(Q, "b c h w -> b (h w) c").contiguous()
        K = rearrange(K, "b c h w -> b (h w) c").contiguous()
        V = rearrange(V, "b c h w -> b (h w) c").contiguous()
        query_norm = self.Q_norm(Q)
        key_norm = self.K_norm(K)
        V_norm = self.V_norm(V)

        int_fetats = self.mlp(self.scale_fusion(query_norm, key_norm, V_norm)[0])

        int_fetats = rearrange(int_fetats, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        return int_fetats


class ChannelAwareMixing(nn.Module):
    def __init__(self, channels):
        super(ChannelAwareMixing, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        return x



def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # elif scheme == 'trunc_normal':
        #     trunc_normal_tf_(module.weight, std=.02)
        #     if module.bias is not None:
        #         nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class RWKVB_layers(nn.Module):
    def __init__(self, num_layers, channels, mlp_ratio, drop_path, total_layers):
        super(RWKVB_layers, self).__init__()
        
        self.reduce = nn.Sequential(nn.Conv2d(channels * 16, channels, 1, bias=False),
                                    nn.BatchNorm2d(channels),
                                    nn.ReLU(True))
        dpr = [x.item() for x in torch.linspace(0, drop_path, channels)]

        self.layers = nn.ModuleList([
            RWKVBlock(
                channels=channels,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                total_layers=total_layers,
                layer_id=i
            ) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x, H, W):
        x = self.reduce(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        for layer in self.layers:
            x = layer(x, H, W)
        x = self.norm(x)
        return x

class RWKVBlock(BaseModule):

    def __init__(
            self,
            channels,
            mlp_ratio=4.,
            drop_path=0.,
            # Meta
            total_layers=None,
            layer_id=None,
            **kwargs
    ):
        super().__init__(init_cfg=None)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        from ext.rwkv.cls_backbones.backbones.vrwkv import VRWKV_SpatialMix, VRWKV_ChannelMix
        self.attn = VRWKV_SpatialMix(
            channels,
            n_layer=total_layers,
            layer_id=layer_id,
            shift_mode='q_shift',
            channel_gamma=.25,
            shift_pixel=1,
            init_mode='fancy',
            key_norm=False,
        )
        self.mlp = VRWKV_ChannelMix(
            channels,
            n_layer=total_layers,
            layer_id=layer_id,
            shift_mode='q_shift',
            channel_gamma=.25,
            shift_pixel=1,
            hidden_rate=mlp_ratio,
            init_mode='fancy',
            key_norm=False
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""
        # B, N, C = x.shape
        # x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x), (H, W)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x), (H, W)))
        # x = x.view(B, C, N).permute(0, 2, 1)
        #x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        return x


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    elif act == 'silu':
        layer = nn.SiLU(inplace)
    elif act == 'SELU':
        layer = nn.SELU(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


def scale_size(x, size):
    '''
    scale x to the same size as y
    '''
    y_size = size // 8 + 1

    # if cfg.OPTIONS.TORCH_VERSION >= 1.5:
    x_scaled = torch.nn.functional.interpolate(
        x, size=(y_size, y_size), mode='bilinear',
        align_corners=True)
    # align_corners=False)
    # else:
    #     x_scaled = torch.nn.functional.interpolate(
    #         x, size=y_size, mode='bilinear',
    #         align_corners=align_corners)
    return x_scaled


def ResizeX(x, scale_factor):
    '''
    scale x by some factor
    '''
    # if cfg.OPTIONS.TORCH_VERSION >= 1.5:
    x_scaled = torch.nn.functional.interpolate(
        x, scale_factor=scale_factor, mode='bilinear',
        align_corners=True)
    # align_corners=False, recompute_scale_factor=True)
    # else:
    #     x_scaled = torch.nn.functional.interpolate(
    #         x, scale_factor=scale_factor, mode='bilinear',
    #         align_corners=False)
    return x_scaled


def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)

    # if cfg.OPTIONS.TORCH_VERSION >= 1.5:
    x_scaled = torch.nn.functional.interpolate(
        x, size=y_size, mode='bilinear',
        align_corners=True)
    # align_corners=False)
    # else:
    #     x_scaled = torch.nn.functional.interpolate(
    #         x, size=y_size, mode='bilinear',
    #         align_corners=align_corners)
    return x_scaled


def make_attn_head(in_ch, out_ch):
    bot_ch = 256

    od = OrderedDict([('conv0', nn.Conv2d(in_ch, bot_ch, kernel_size=3,
                                          padding=1, bias=False)),
                      ('bn0', nn.BatchNorm2d(bot_ch)),
                      ('re0', nn.ReLU(inplace=True))])

    od['conv1'] = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1,
                            bias=False)
    od['bn1'] = nn.BatchNorm2d(bot_ch)
    od['re1'] = nn.ReLU(inplace=True)
    od['conv2'] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od['sig'] = nn.Sigmoid()
    attn_head = nn.Sequential(od)
    # init_attn(attn_head)
    return attn_head


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
