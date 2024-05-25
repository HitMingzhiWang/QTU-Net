import torch
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple

import torch.nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:  # drop_prob废弃率=0，或者不是训练的时候，就保持原来不变
        return x
    keep_prob = 1 - drop_prob  # 保持率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (b, 1, 1, 1) 元组  ndim 表示几维，图像为4维
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 0-1之间的均匀分布[2,1,1,1]
    random_tensor.floor_()  # 下取整从而确定保存哪些样本 总共有batch个数
    output = x.div(keep_prob) * random_tensor  # 除以 keep_prob 是为了让训练和测试时的期望保持一致
    # 如果keep，则特征值除以 keep_prob；如果drop，则特征值为0
    return output  # 与x的shape保持不变


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(eps=1e-5, num_features=out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(eps=1e-5, num_features=out_channels)
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_features=dim, eps=1e-5)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(num_features=dim, eps=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(num_features=embed_dim, eps=1e-5)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class UplapPatchEmed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       )
        self.norm = nn.BatchNorm2d(num_features=embed_dim, eps=1e-5)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class SegNext_Unet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 num_classes=2
                 # pretrained=None,
                 # init_cfg=None
                 ):
        super(SegNext_Unet, self).__init__()

        # assert not (init_cfg and pretrained), \
        #     'init_cfg and pretrained cannot be set at the same time'
        # if isinstance(pretrained, str):
        #     warnings.warn('DeprecationWarning: pretrained is deprecated, '
        #                   'please use "init_cfg" instead')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        # elif pretrained is not None:
        #     raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                )

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         )
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        cur = cur - depths[num_stages - 1]
        for k in range(num_stages):
            print(num_stages - k - 1)
            if k == num_stages - 1:
                upPatch_embed = StemConv(embed_dims[0], num_classes)
            else:
                upPatch_embed = UplapPatchEmed(patch_size=7 if k == num_stages - 1 else 3,
                                                stride=4 if k == num_stages - 1 else 2,
                                                in_chans=in_chans if k == num_stages - 1 else embed_dims[
                                                    num_stages - k - 1],
                                                embed_dim=embed_dims[num_stages - k - 2],
                                                )
                upBlock = nn.ModuleList(
                    [Block(dim=embed_dims[num_stages - k - 2], mlp_ratio=mlp_ratios[num_stages - k - 2],
                           drop=drop_rate, drop_path=dpr[cur + j],
                           )
                     for j in range(depths[num_stages - k - 2])])
                upNorm = nn.LayerNorm(embed_dims[num_stages - k - 2])
                cur -= depths[num_stages - k - 2]
                setattr(self, f"upPatch_embed{k + 1}", upPatch_embed)
                setattr(self, f"upBlock{k + 1}", upBlock)
                setattr(self, f"upNorm{k + 1}", upNorm)

    # def init_weights(self):
    #     print('init cfg', self.init_cfg)
    #     if self.init_cfg is None:
    #         for m in self.modules():
    #             if isinstance(m, nn.Linear):
    #                 trunc_normal_init(m, std=.02, bias=0.)
    #             elif isinstance(m, nn.LayerNorm):
    #                 constant_init(m, val=1.0, bias=0.)
    #             elif isinstance(m, nn.Conv2d):
    #                 fan_out = m.kernel_size[0] * m.kernel_size[
    #                     1] * m.out_channels
    #                 fan_out //= m.groups
    #                 normal_init(
    #                     m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
    #     else:
    #
    #         super(MSCAN, self).init_weights()

    def forward(self, x):
        B = x.shape[0]
        features = []  # 相当于Unet中的各层级Features map

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            print(x.shape)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            print(x.shape)
            features.append(x)

        for k in range(self.num_stages):
            if k != self.num_stages - 1:
                upPatch_embed = getattr(self, f"upPatch_embed{k + 1}")
                upBlock = getattr(self, f"upBlock{k + 1}")
                upNorm = getattr(self, f"upNorm{k + 1}")
                x, H, W = upPatch_embed(x)
                for blk in upBlock:
                    x = blk(x, H, W)
                x = upNorm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


x = torch.randn(1, 3, 256, 256)
model = SegNext_Unet(embed_dims=[32, 64, 128, 256],
                     mlp_ratios=[8, 8, 4, 4],
                     drop_rate=0.0,
                     drop_path_rate=0.1,
                     num_stages=4,
                     depths=[2, 2, 2, 2],
                     num_classes=2
                     )

x = model(x)


