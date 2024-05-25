import copy

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn


from PIL import Image
from scipy.spatial.distance import directed_hausdorff


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_indicators(predictions, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    predictions = predictions.flatten()
    labels = labels.flatten()
    for i in range(len(predictions)):
        if predictions[i] == 1 and labels[i] == 1:
            tp += 1
        elif predictions[i] == 1 and labels[i] == 0:
            fp += 1
        elif predictions[i] == 0 and labels[i] == 0:
            tn += 1
        elif predictions[i] == 0 and labels[i] == 1:
            fn += 1

    if (
            tp + fn == 0 or tp + fp == 0 or tp + fn + fp == 0 or tp == 0 or tp + fp + fn == 0 or tp + tn == 0 or tp + fp + fn + tn == 0):
        com = 0
        cor = 0
        q = 0
        f1 = 0
        iou = 0
        acc = 0
    else:
        com = tp / (tp + fn)
        cor = tp / (tp + fp)
        q = tp / (tp + fn + fp)
        f1 = 2 * com * cor / (com + cor)
        iou = tp / (tp + fp + fn)
        acc = (tp + tn) / (tp + fp + fn + tn)
    return com, cor, q, f1, iou, acc


# def compute_hd95(pred, label):
#     # pred = pred.squeeze().cpu().numpy()
#     # label = label.squeeze().cpu().numpy()
#     distances = []
#     for i in range(pred.shape[0]):
#         d = directed_hausdorff(pred[i], label[i])[0]
#         distances.append(d)
#     distances = np.array(distances)
#     hd95 = np.percentile(distances, 95)
#     return hd95


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        com, cor, q, f1,iou,acc = calculate_indicators(pred, gt)
        return dice, hd95, com, cor, q, f1,iou,acc
    elif pred.sum() > 0 and gt.sum() == 0:
        com, cor, q, f1,iou,acc = calculate_indicators(pred, gt)
        return 1, 0, com, cor, q, f1,iou,acc
    else:
        com, cor, q, f1,iou,acc = calculate_indicators(pred, gt)
        return 0, 0, com, cor, q, f1,iou,acc


# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
#             if x != patch_size[0] or y != patch_size[1]:
#                 slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
#             input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
#             net.eval()
#             with torch.no_grad():
#                 outputs = net(input)
#                 out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().numpy()
#                 if x != patch_size[0] or y != patch_size[1]:
#                     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 else:
#                     pred = out
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().numpy()
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))
#
#     if test_save_path is not None:
#         img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#         prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#         lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#         img_itk.SetSpacing((1, 1, z_spacing))
#         prd_itk.SetSpacing((1, 1, z_spacing))
#         lab_itk.SetSpacing((1, 1, z_spacing))
#         sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
#         sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
#         sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
#     return metric_list


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    _, x, y = image.shape

    # 缩放图像符合网络输入大小224x224
    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        # 缩放预测结果图像同原始图像大小
        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        a1 = copy.deepcopy(prediction)
        a2 = copy.deepcopy(prediction)
        a3 = copy.deepcopy(prediction)
        # r通道
        a1[a1 == 1] = 255
        # g通道
        a2[a2 == 1] = 255
        # b通道
        a3[a3 == 1] = 255
        a1 = Image.fromarray(np.uint8(a1)).convert('L')
        a2 = Image.fromarray(np.uint8(a2)).convert('L')
        a3 = Image.fromarray(np.uint8(a3)).convert('L')
        prediction = Image.merge('RGB', [a1, a2, a3])
        prediction.save(test_save_path + '/' + case + '.png')
    return metric_list
