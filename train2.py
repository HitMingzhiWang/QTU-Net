import argparse
import os
import random
import sys
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from networks.Segnet.Segnet import SegNet
from trainer2 import trainer_synapse
from config import get_config
# from networks.MECNet.MECNet import MECNet
from networks.QTNUnet.qtnUnet import qtnUnet
from networks.UnetPlusPlus.UnetPlusPlus import NestedUNet
from networks.FCN.fcn_model import fcn_resnet50, FCN
from networks.Attenion_Unet.Attention_Unet import AttU_Net
from networks.R2Unet.R2Unet import R2U_Net
from networks.DSUnet3.vit_seg_configs import get_r50_b16_config
from networks.DSUnet3.Model import VisionTransformer
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.SKUnet.SKUnet import SKUnet
from networks.Unet.Unet import Unet
# from networks.QTNUnet.Unet2 import Unet2
from networks.TransUnet.vit_seg_modeling import VisionTransformer
from networks.mfSegformer.segformer import SegFormer
from networks.DeeplabV3.deeplabv3 import DeepLabV3Plus
import networks.TransUnet.vit_seg_configs as configs
from networks.Segnet.Segnet import SegNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./datasets/WaterBodies/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='WHDLD', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_WaterBodies', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--output_dir', default='./output', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=32, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default=r'configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE",
                    help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Palsr': {
            'root_path': './datasets/Palsr/train_npz',
            'list_dir': './lists/lists_Palsr',
            'num_classes': 2,
        },
        'Sen': {
            'root_path': './datasets/Sen/train_npz',
            'list_dir': './lists/lists_Sen',
            'num_classes': 2,
        },
        'WHDLD': {
            'root_path': './datasets/WHDLD/train_npz',
            'list_dir': './lists/lists_WHDLD',
            'num_classes': 2,
        },
        'WaterBodies': {
            'root_path': './datasets/WaterBodies/train_npz',
            'list_dir': './lists/lists_WaterBodies',
            'num_classes': 2,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # net = DeepLabV3Plus(num_classes=2).cuda()
    # net = SegFormer(num_classes = 2 , phi = 'b0', pretrained = False).cuda()
    # net = MECNet().cuda()
    # net = Resnet_Unet(num_classes=2).cuda()
    # net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    # net = SegNet(in_channels=3, num_classes=2).cuda()
    # net = qtnUnet(img_size=256, in_channel=3, depth=[2, 2, 2, 2, 2], embed_dim=[16, 32, 64, 128, 256],
    #               num_classes=2).cuda()
    # net = Unet2(img_size=256, in_channel=3, depth=[2, 2, 2, 2, 2], embed_dim=[16, 32, 64, 128, 256],
    #               num_classes=2).cuda()
    # net = qtnUnet(img_size=256, in_channel=3, depth=[1, 1, 1, 1, 1], embed_dim=[16, 32, 64, 128, 256],
    #               num_classes=2).cuda()
    # net = qtnUnet(img_size=256, in_channel=3, depth=[2, 2, 2, 2, 2], embed_dim=[32, 64, 128, 256, 512],
    #               num_classes=2).cuda()
    # net = qtnUnet(img_size=256, in_channel=3, depth=[1, 1, 1, 2, 2], embed_dim=[16, 32, 64, 128, 256],
    #               num_classes=2).cuda()
    # net.load_from(config)
    # net = AttU_Net(in_channel=3,num_classes=1,checkpoint=False,channel_list=[64, 128, 256, 512, 1024],convTranspose=True).cuda()
    # net = fcn_resnet50(num_classes=2,aux=False ).cuda()
    # config1 = get_r50_b16_config()
    # net = qtnUnet(img_size=256, in_chans=3, embed_dims=[16, 64, 128, 256], mlp_ratios=[4, 4, 4, 4],
    #                 norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 1, 1, 1], ).cuda()
    # net = NestedUNet(input_channels=3,t=2,num_classes=2).cuda()
    # net = SegNet(in_channels=3, num_classes=2).cuda()
    # net = R2U_Net(img_ch=3,output_ch=2,t=2).cuda()
    net = Unet(in_channels=3, classes=2).cuda()
    # net_dict = net.state_dict()
    # pretrained_dict = torch.load('output/epoch_399.pth')
    # net_dict.update(pretrained_dict)
    # net.load_state_dict(net_dict)
    # config = configs.get_r50_b16_config()
    # net = VisionTransformer(config=config, img_size=256, num_classes=2, zero_head=False, vis=False).cuda()
    # net = SKUnet(img_size=256, in_chans=3, embed_dims=[16, 64, 128, 256], mlp_ratios=[4, 4, 4, 4],norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2]).cuda()
    trainer = {'WHDLD': trainer_synapse, }
    trainer[dataset_name](args, net, args.output_dir)
