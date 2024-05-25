import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import Dropout

from networks.DSUnet2.vit_seg_modeling_resnet_skip import ResNetV2
from networks.DSUnet2.vit_seg_configs import get_r50_b16_config
from timm.models.layers import trunc_normal_, DropPath, to_2tuple



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=3072, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Position(nn.Module):
    def __init__(self,final_dim,nums):
        self.position_embeddings = nn.Parameter(torch.zeros(1,nums,final_dim))
        self.dropout = Dropout(0.1)
    def forward(self,x):
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings






class DsAttention(nn.Module):
    def __init__(self, dim=48, n_heads=3, dropout=None, m_power=None,in_channels=1024):
        super(DsAttention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.l_q = nn.Linear(dim, dim)
        self.l_k = nn.Linear(dim, dim)
        self.l_v = nn.Linear(dim, dim)
        self.g_q = nn.Linear(dim, dim)
        self.g_k = nn.Linear(dim, dim)
        self.g_v = nn.Linear(dim, dim)
        self.m_power = m_power
        self.scale = torch.sqrt(torch.cuda.FloatTensor([self.dim // self.n_heads]))
        self.atten_drop = nn.Dropout(dropout)
        # self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=1024,out_channels=dim,kernel_size=1)
        self.conMerge = nn.Conv2d(in_channels=2*dim,out_channels=dim,kernel_size=1)
        self.act = nn.ReLU()
    def forward(self, x,feature):
        x1 = x
        x1 = self.conv1(x1)
        x2 = feature
        x1 = rearrange(x1, "b c h w -> b (h w) c").contiguous()
        x2 = rearrange(x2, "b c h w -> b (h w) c").contiguous()
        Batch = x1.shape[0]
        L_Q = self.l_q(x1)
        L_K = self.l_k(x1)
        L_V = self.l_v(x1)
        G_Q = self.g_q(x2)
        G_K = self.g_k(x2)
        G_V = self.g_v(x2)
        L_Q = L_Q.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        L_K = L_K.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        L_V = L_V.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        G_Q = G_Q.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        G_K = G_K.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        G_V = G_V.view(Batch, -1, self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        attentionLG = (L_Q @ G_K.transpose(-2, -1)) / self.scale
        attentionGL = (G_Q @ L_K.transpose(-2, -1)) / self.scale
        attentionLG = self.atten_drop(torch.softmax(attentionLG, dim=-1))
        attentionGL = self.atten_drop(torch.softmax(attentionGL, dim=-1))
        x1 = attentionLG @ G_V
        x2 = attentionGL @ L_V
        x1 = x1.permute(0, 2, 1, 3).contiguous()
        x2 = x2.permute(0, 2, 1, 3).contiguous()
        x1 = x1.view(Batch, -1, self.n_heads * (self.dim // self.n_heads))
        x2 = x2.view(Batch, -1, self.n_heads * (self.dim // self.n_heads))
        h1 = int(math.sqrt(x1.shape[1]))
        h2 = int(math.sqrt(x2.shape[1]))
        x1 = rearrange(x1, "b (h w) c -> b h w c", w=h1, h=h1).contiguous().permute(0, 3, 1, 2)
        x2 = rearrange(x2, "b (h w) c -> b h w c", w=h2, h=h2).contiguous().permute(0, 3, 1, 2)
        if self.m_power != None:
            x1 = nn.Upsample(scale_factor=self.m_power)(x1)
        x1 = torch.cat([x1,x2],dim=1)
        x1 = self.conMerge(x1)
        x1 = self.act(x1)

        return x1















class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        # B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # x = hidden_states.permute(0, 2, 1)
        # x = x.contiguous().view(B, hidden, h, w)
        x = hidden_states
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):  # 从这往下没看懂
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.Encoder = ResNetV2(width_factor=1, block_units=[3, 4, 9])
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.decoder = DecoderCup(config)
        self.dsAttention3 = DsAttention(dim=64, n_heads=2, dropout=0.05, m_power=8,in_channels=1024)
        self.dsAttention2 = DsAttention(dim=256, n_heads=2, dropout=0.05, m_power=4,in_channels=1024)
        self.dsAttention1 = DsAttention(dim=512, n_heads=2, dropout=0.05, m_power=2,in_channels=1024)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        self.tranformer = Block(dim=1024, num_heads=8,mlp_ratio=3., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.Encoder(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        x = self.tranformer(x)
        x = rearrange(x, "b (h w) c -> b h w c", w=16, h=16).contiguous().permute(0, 3, 1, 2)
        features2 = []
        features2.append(self.dsAttention1(x, features[0]))
        features2.append(self.dsAttention2(x, features[1]))
        features2.append(self.dsAttention3(x, features[2]))
        x = self.decoder(x, features2)
        logits = self.segmentation_head(x)
        return logits

