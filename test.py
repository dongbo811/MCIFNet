import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import test_dataset
 

def object(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the object score.
        """
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * s_object(fg, gt) + (1 - u) * s_object(bg, 1 - gt)
        return object_score

def s_object(pred: np.ndarray, gt: np.ndarray) -> float:
    x = np.mean(pred[gt == 1])
    sigma_x = np.std(pred[gt == 1])
    score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
    return score

def region( pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate the region score.
    """
    x, y = centroid(gt)
    part_info = divide_with_xy(pred, gt, x, y)
    w1, w2, w3, w4 = part_info["weight"]
    # assert np.isclose(w1 + w2 + w3 + w4, 1), (w1 + w2 + w3 + w4, pred.mean(), gt.mean())

    pred1, pred2, pred3, pred4 = part_info["pred"]
    gt1, gt2, gt3, gt4 = part_info["gt"]
    score1 = ssim(pred1, gt1)
    score2 = ssim(pred2, gt2)
    score3 = ssim(pred3, gt3)
    score4 = ssim(pred4, gt4)

    return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

def centroid(matrix: np.ndarray) -> tuple:
    """
    To ensure consistency with the matlab code, one is added to the centroid coordinate,
    so there is no need to use the redundant addition operation when dividing the region later,
    because the sequence generated by ``1:X`` in matlab will contain ``X``.

    :param matrix: a data array
    :return: the centroid coordinate
    """
    h, w = matrix.shape
    if matrix.sum() == 0:
        x = np.round(w / 2)
        y = np.round(h / 2)
    else:
        area_object = np.sum(matrix)
        row_ids = np.arange(h)
        col_ids = np.arange(w)
        x = np.round(np.sum(np.sum(matrix, axis=0) * col_ids) / area_object)
        y = np.round(np.sum(np.sum(matrix, axis=1) * row_ids) / area_object)
    return int(x) + 1, int(y) + 1

def divide_with_xy( pred: np.ndarray, gt: np.ndarray, x: int, y: int) -> dict:
    """
    Use (x,y) to divide the ``pred`` and the ``gt`` into four submatrices, respectively.
    """
    h, w = gt.shape
    area = h * w

    gt_LT = gt[0:y, 0:x]
    gt_RT = gt[0:y, x:w]
    gt_LB = gt[y:h, 0:x]
    gt_RB = gt[y:h, x:w]

    pred_LT = pred[0:y, 0:x]
    pred_RT = pred[0:y, x:w]
    pred_LB = pred[y:h, 0:x]
    pred_RB = pred[y:h, x:w]

    w1 = x * y / area
    w2 = y * (w - x) / area
    w3 = (h - y) * x / area
    w4 = 1 - w1 - w2 - w3

    return dict(
        gt=(gt_LT, gt_RT, gt_LB, gt_RB),
        pred=(pred_LT, pred_RT, pred_LB, pred_RB),
        weight=(w1, w2, w3, w4),
    )

def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate the ssim score.
    """
    h, w = pred.shape
    N = h * w

    x = np.mean(pred)
    y = np.mean(gt)

    sigma_x = np.sum((pred - x) ** 2) / (N - 1)
    sigma_y = np.sum((gt - y) ** 2) / (N - 1)
    sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

    if alpha != 0:
        score = alpha / (beta + _EPS)
    elif alpha == 0 and beta == 0:
        score = 1
    else:
        score = 0
    return score


_EPS = 1e-16
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--pth_path', type=str, default='../pth/submit/MCIFNet-best.pth')
opt = parser.parse_args()
# 'CAMO',COD10K
for _data_name in [ 'CAMO']:
    data_path = './demodata/{}/'.format(_data_name)
    
    model = Network(channel=32)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    # os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    MAE = 0
    SM = 0
    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res = model(image)
        # print(gt.shape)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # print('> {} - {}'.format(_data_name, name))

        # cv2.imwrite(save_path+name,res*255)

        input = res
        pred = res
        
        # sm
        mask = gt
        MAE += np.mean(np.abs(mask - pred))
        mask, pred = (mask.squeeze(), pred.squeeze())
        # print(mask.shape, pred.shape)
        y = np.mean(mask)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            alpha = 0.5
            sm = alpha * object(pred, mask) + (1 - alpha) * region(pred, mask)
            sm = max(0, sm)

            SM += sm
            
    print("dataset: ", _data_name, 'SM:', SM/test_loader.size, 'MAE: ', MAE/test_loader.size)

           