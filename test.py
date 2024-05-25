import argparse
import logging
import os
import random
import sys
from functools import partial
from networks.QTNUnet.qtnUnet import qtnUnet
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.mfSegformer.segformer import SegFormer
from networks.Unet.Unet import Unet
from networks.FCN.fcn_model import fcn_resnet50, FCN
from datasets.dataset_synapse import Synapse_dataset
from networks.DeeplabV3.deeplabv3 import DeepLabV3Plus
from networks.Segnet.Segnet import SegNet
from utils import test_single_volume
from config import get_config

from networks.vision_transformer import SwinUnet as ViT_seg
from networks.UnetPlusPlus.UnetPlusPlus import NestedUNet
from networks.Attenion_Unet.Attention_Unet import AttU_Net
from networks.R2Unet.R2Unet import R2U_Net
# from networks.DSUnet3.vit_seg_configs import get_r50_b16_config
# from networks.DSUnet3.Model import VisionTransformer

# from networks.TransUnet.vit_seg_modeling import VisionTransformer
import networks.TransUnet.vit_seg_configs as configs

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./datasets/GLH-Water',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='WHDLD', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_GLH-Water', help='list dir')
parser.add_argument('--output_dir', default='./output', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True, action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
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
if args.dataset == "GLH-Water":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_com %f mean_cor %f mean_q %f mean_f1 %f mean_iou %f mean_acc %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],np.mean(metric_i,axis=0)[2],np.mean(metric_i,axis=0)[3],np.mean(metric_i,axis=0)[4],np.mean(metric_i,axis=0)[5],np.mean(metric_i,axis=0)[6],np.mean(metric_i,axis=0)[7]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f mean_com %f mean_cor %f mean_q %f mean_f1 %f mean_iou %f mean_acc %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2], metric_list[i - 1][3], metric_list[i - 1][4], metric_list[i - 1][5], metric_list[i - 1][6], metric_list[i - 1][7]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    com = np.mean(metric_list, axis=0)[2]
    cor = np.mean(metric_list, axis=0)[3]
    q = np.mean(metric_list, axis=0)[4]
    f1 = np.mean(metric_list, axis=0)[5]
    iou = np.mean(metric_list, axis=0)[6]
    acc = np.mean(metric_list, axis=0)[7]
    logging.info('Testing performance in best val model: mean_dice %f mean_hd95 %f mean_com %f mean_cor %f mean_q %f mean_f1 %f mean_iou %f mean_acc %f mean %f' % (performance, mean_hd95,com,cor,q,f1,iou,acc,(performance+com+cor+q+f1+iou+acc)/7))
    return "Testing Finished!"


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

    dataset_config = {
        'Palsr': {
            'Dataset': Synapse_dataset,
            'volume_path': './datasets/Palsr/test_vol_h5',
            'list_dir': './lists/lists_Palsr',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'Sen': {
            'Dataset': Synapse_dataset,
            'volume_path': './datasets/Sen/test_vol_h5',
            'list_dir': './lists/lists_Sen',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'WHDLD': {
            'Dataset': Synapse_dataset,
            'volume_path': './datasets/WHDLD/test_vol_h5',
            'list_dir': './lists/lists_WHDLD',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'WaterBodies': {
            'Dataset': Synapse_dataset,
            'volume_path': './datasets/WaterBodies/test_vol_h5',
            'list_dir': './lists/lists_WaterBodies',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'GLH-Water': {
            'Dataset': Synapse_dataset,
            'volume_path': './datasets/GLH-Water/test_vol_h5',
            'list_dir': './lists/lists_GLH-Water',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    config1 = configs.get_r50_b16_config()
    net = Unet(in_channels=3,classes=2).cuda()
    # net = qtnUnet(img_size=256, in_channel=3, depth=[2, 2, 2, 2, 2], embed_dim=[16, 32, 64, 128, 256],
    #               num_classes=2).cuda()
    # net = qtnUnet(img_size=256, in_channel=3, depth=[1, 1, 1, 1, 1], embed_dim=[16, 32, 64, 128, 256],
    #               num_classes=2).cuda()
    # net = VisionTransformer(num_classes=2, config=config1,img_size=256).cuda()
    # net = ViT_seg(config=config,img_size=512,num_classes=2).cuda()
    # net = AttU_Net(in_channel=3, num_classes=2, checkpoint=False, channel_list=[64, 128, 256, 512, 1024]
    #                ,convTranspose=True).cuda()
    # net = NestedUNet(num_classes=2, input_channels=3).cuda()
    # net = Unet(classes=2, in_channels=3).cuda()
    # net = DeepLabV3Plus(num_classes=2).cuda()
    # net = SegFormer(num_classes = 2 , phi = 'b0', pretrained = False).cuda()
    # net = fcn_resnet50(num_classes=2, aux=False).cuda()
    # net = R2U_Net(img_ch=3, output_ch=2, t=2).cuda()
    # net = SKUnet(img_size=256, in_chans=3, embed_dims=[16, 64, 128, 256], mlp_ratios=[4, 4, 4, 4],
    #              norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2]).cuda()
    # net = SegNet(in_channels=3, num_classes=2).cuda()
    # config = configs.get_r50_b16_config()
    # net = UnetQuaterion(in_channels=3, classes=2).cuda()
    # net = VisionTransformer(config=config1, img_size=256, num_classes=2, zero_head=False, vis=False).cuda()
    snapshot = os.path.join(args.output_dir, 'epoch_199.pth')

    # device = {'cuda:0': 'cuda:0', 'cuda:1' : 'cuda:1','cuda:2': 'cuda:2','cuda:3': 'cuda:3'}
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))
    # msg = net.load_state_dict(
    #     {k.replace('module.', ''): v for k, v in torch.load(snapshot, map_location=device).items()},
    #     strict=True)
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet", msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
    # for i in range(20, 40):
    #     str1 = "epoch_" + str(i *10) + ".pth"
    #     snapshot = os.path.join(args.output_dir, str1)
    #     msg = net.load_state_dict(torch.load(snapshot))
    #     print("self trained swin unet", msg)
    #     logging.info(snapshot)
    #     net.load_state_dict(torch.load(snapshot))
    #     inference(args, net, test_save_path)