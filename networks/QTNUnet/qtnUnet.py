import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
# from .core.quaternion_layers import *
from .core.quaternion_layers import *
from QCNN.generateq import *
import sys


class DWConv(nn.Module):
    def __init__(self, dim=256):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, padding=1, bias=True)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = QuaternionConv(in_features, hidden_features, 1, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = QuaternionConv(hidden_features, out_features, 1, 1)
        # self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x

class SKmodule(nn.Module):
    def __init__(self,dim):
        super(SKmodule, self).__init__()
        self.conv0_1 = nn.Conv2d(dim,dim,kernel_size=(7,1),padding=(0,3))
        self.conv0_2 = nn.Conv2d(dim,dim,kernel_size=(1,7),padding=(3,0))
        self.conv1_1 = nn.Conv2d(dim,dim,kernel_size=3,padding=1)
        self.conv1_2 = nn.Conv2d(dim,dim,kernel_size=5,padding=4,dilation=2)
        self.conv1_3 = nn.Conv2d(dim,dim,kernel_size=7,padding=9,dilation=3)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)



    def forward(self,x):
        attn1 = self.conv0_1(x)
        attn1 = self.conv0_2(attn1)

        attn2 = self.conv1_1(x)
        attn2 = self.conv1_2(attn2)
        attn2 = self.conv1_3(attn2)

        attn = torch.cat([attn1,attn2],dim=1)

        avg_attn = torch.mean(attn,dim=1,keepdim=True)
        max_attn, _ = torch.max(attn,dim=1,keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = avg_attn * sig[:, 0, :, :].unsqueeze(1) + max_attn * sig[:, 1, :, :].unsqueeze(1)
        return x * attn


class SKConv(nn.Module):
    def __init__(self, features, M, G, r, stride=1, L=16):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.conv0_1 = QuaternionConv(features, features, kernel_size=(7, 1), padding=(0, 3))
        self.conv0_2 = QuaternionConv(features, features, kernel_size=(1, 7), padding=(3, 0))
        self.conv1_1 = QuaternionConv(features, features, kernel_size=3, padding=1)
        self.conv2_1 = QuaternionConv(features, features, kernel_size=7, padding=9, dilation=3)


        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        fea_0 = self.conv0_1(self.conv0_2(x)).unsqueeze(dim=1)
        fea_1 = self.conv1_1(x).unsqueeze(dim=1)
        fea_2 = self.conv2_1(x).unsqueeze(dim=1)
        feas = torch.cat([fea_0,fea_1,fea_2],dim=1)

        fea_U = torch.sum(feas, dim=1)

        fea_s = fea_U.mean(-1).mean(-1)

        fea_z = self.fc(fea_s)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v





class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = QuaternionConv(dim, dim, 3, padding=1, stride=1)
        # self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, stride=1)
        # self.conv_spatial = nn.Conv2d(dim, dim, 3, stride=1, padding=3, dilatation=3)
        self.conv_spatial = QuaternionConv(2 * dim, dim, 3, stride=1, padding=3, dilatation=3)
        # self.conv_spatial = nn.Conv2d(2 * dim, dim, 3, stride=1, padding=1)
        self.conv1 = QuaternionConv(dim, dim, 1, 1)
        # self.conv1 = nn.Conv2d(dim, dim, 1, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(torch.cat([attn, attn], 1))
        # attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        attn = u * attn

        return attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()


        self.proj_1 = QuaternionConv(d_model, d_model, kernel_size=3, stride=1,padding=1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        # self.spatial_gating_unit = SKConv(features=d_model,M=3,G=1,r=8,L=16)
        # self.spatial_gating_unit = SKmodule(dim=d_model)
        # self.proj_2 = nn.Conv2d(d_model, d_model, 1, 1)
        self.proj_2 = QuaternionConv(d_model, d_model, 1, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        # self.norm1 = QuaternionBatchNorm2d(dim, gamma_init=1.0, beta_param=True)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        # self.norm2 = QuaternionBatchNorm2d(dim, gamma_init=1.0, beta_param=True)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class UpSampleEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=3, stride=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.UpsamplingBilinear2d(scale_factor=2)
        self.covUp = nn.Conv2d(kernel_size=1,in_channels=in_chans,out_channels=embed_dim)
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=stride,  padding=0)
        # self.norm = QuaternionBatchNorm2d(embed_dim)
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = self.covUp(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=3, stride=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=stride,  padding=0)
        # self.norm = QuaternionBatchNorm2d(embed_dim)
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_size=256, in_chans=3, depths=[3, 4, 6, 3], num_stages=4, drop_rate=0., drop_path_rate=0.,
                 stride=[2, 2, 2, 2], mlp_ratios=[4, 4, 4, 4], embed_dims=[64, 128, 256, 512],
                 norm_layer=nn.LayerNorm, ):
        super().__init__()
        self.conExpand = nn.Conv2d(in_channels=in_chans,kernel_size=1,out_channels=16)
        self.pre_press = generateq(channel=16, k_size=1)
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=3 if i == 0 else 3,
                                            # stride=2 if i == 0 else 2,
                                            stride=stride[i],
                                            in_chans=4 if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward_features(self, x):
        B = x.shape[0]
        # x = patch_embed(x)
        featureMap = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x = patch_embed(x)
            _, _, H, W = x.shape
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i != self.num_stages - 1:
                featureMap.append(x)
        return x, featureMap

    def forward(self, x):
        # x = x.squeeze()
        x = self.conExpand(x)
        x = self.pre_press(x)
        # x = self.forward_embeddings(x)
        x, featureMap = self.forward_features(x)
        return x, featureMap


class Decoder(nn.Module):
    def __init__(self, img_size=256, depths=[3, 4, 6, 3], num_stages=3, drop_rate=0., drop_path_rate=0.,
                 stride=[2, 2, 2, 2], mlp_ratios=[4, 4, 4, 4], embed_dims=[256,128, 64, 16],
                 norm_layer=nn.LayerNorm, ):
        super(Decoder, self).__init__()
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            patch_embed = UpSampleEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=3 if i == 0 else 3,
                                            # stride=2 if i == 0 else 2,
                                            stride=stride[i],
                                            in_chans=256 if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            # else:
            #     patch_embed = nn.Sequential(
            #         nn.Conv2d(in_channels=embed_dims[i-1],out_channels=embed_dims[i],kernel_size=3,padding=1),
            #         nn.MaxPool2d(2)
            #     )

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            covCat = QuaternionConv(in_channels=embed_dims[i] * 2, out_channels=embed_dims[i], stride=1, kernel_size=3,padding=1)
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
            setattr(self, f"covCat{i + 1}", covCat)

    def forward(self, x,featureMap):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            covCat = getattr(self, f"covCat{i + 1}")
            x = patch_embed(x)
            if x.shape[2] != featureMap[self.num_stages - i - 1].shape[2]:
                paddingNum = featureMap[self.num_stages - i - 1].shape[2] - x.shape[2]
                pad = nn.ZeroPad2d(padding=(paddingNum,0,paddingNum,0))
                x = pad(x)
            x = torch.cat([x, featureMap[self.num_stages - i - 1]], dim=1)
            x = covCat(x)
            _, _, H, W = x.shape
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class SegmentationHead(nn.Module):
    def __init__(self,num_classes=2,inchannels=16):
        super(SegmentationHead, self).__init__()
        self.upCov = nn.UpsamplingBilinear2d(scale_factor=2)

        self.cov = nn.Conv2d(in_channels=inchannels,out_channels=num_classes,kernel_size=1,padding=0,stride=1)
    def forward(self,x):
        x = self.cov(x)
        x = self.upCov(x)
        return x

class qtnUnet(nn.Module):
    def __init__(self,img_size=256, in_chans=3, embed_dims=[16, 64, 128, 256], mlp_ratios=[8, 8, 4, 4],
                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],):
        super(qtnUnet, self).__init__()
        self.encoder = Encoder(img_size=256, in_chans=3, embed_dims=embed_dims, mlp_ratios=[4, 4, 4, 4],
                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths,)
        self.decoder = Decoder(img_size=16, embed_dims=[128, 64, 16], mlp_ratios=[4, 4, 4, 4],
                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=depths)
        self.segmentationHead = SegmentationHead(num_classes=2,inchannels=16)

    def forward(self,x):
        x,featureMap = self.encoder(x)
        x = self.decoder(x,featureMap)
        x = self.segmentationHead(x)
        return x


model = qtnUnet(img_size=256, in_chans=3, embed_dims=[16, 64, 128, 256], mlp_ratios=[8, 8, 4, 4],
                norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], )
x = torch.rand(2, 3, 256, 256)
print(model(x).shape)