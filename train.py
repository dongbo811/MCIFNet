
import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import get_loader, val_loader
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn

_EPS = 1e-16
def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()

            preds = model(images)
            # loss_init = structure_loss(preds[0], gts) + structure_loss(preds[1], gts) + structure_loss(preds[2], gts)
            loss_final = structure_loss(preds, gts)
            loss_init = loss_final
            loss = loss_final

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} '
                    'Loss2: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.data, 'Loss_final': loss_final.data,
                                    'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                # TensorboardX-Outputs
                res = preds[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_init', torch.tensor(res), step, dataformats='HW')
                res = preds[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise

    global dict_plot
    global bestdice
    global best_epoch
    global bestmae

    test1path ='/mnt/workspace/dongbo/ais/dataset/dataset/demodata/'
    if (epoch + 1) % 1 == 0:
        for dataset in ['CAMO']: 
            dataset_dice, mae = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            # print(dataset, ': ', dataset_dice)
            dict_plot[dataset] = dataset_dice
        # print(type(dict_plot['CAMO']))
        mDice = dict_plot['CAMO'] 

        print('Epoch: {}, mDice: {}, bestDICE: {}, bestEpoch: {}, bestmae: {}.'.format(epoch, mDice, bestdice, best_epoch, mae))
        if mDice > bestdice:
            bestdice = mDice
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'MCIFNet.pth')
            torch.save(model.state_dict(), save_path + str(epoch) + 'MCIFNet-best.pth')
            print('Save state_dict successfully! Best epoch:{}.'.format(epoch))

        logging.info('[Val Info]:Epoch:{} DICE:{} bestEpoch:{} bestdice:{}, bestmae:{}'.format(epoch, mDice, best_epoch, bestdice, mae))
        
#         if mae < bestmae:
#             bestmae = mae
#             # best_epoch = epoch
#             # torch.save(model.state_dict(), save_path + 'MCIFNet.pth')
#             # torch.save(model.state_dict(), save_path + str(epoch) + 'MCIFNet-best.pth')
#             # print('Save state_dict successfully! Best epoch:{}.'.format(epoch))

#         logging.info('[Val Info]:Epoch:{} DICE:{} bestEpoch:{} bestmae:{}'.format(epoch, mDice, best_epoch, bestmae))

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
def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    model.eval()


    test_loader = val_loader(image_root=image_root,
                              gt_root=gt_root,
                              batchsize=1,
                              trainsize=384,
                              num_workers=8)
    num1 = len(test_loader)
    MAE = 0.0
    SM = 0.0
    for i, (image, gt) in enumerate(test_loader):
        # image, gt, name, img_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res= model(image)
        # eval Dice
        res = F.upsample(res , size=(gt.shape[2],gt.shape[3]), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        pred = res
        
        # sm
        mask = gt
        MAE += np.mean(np.abs(mask - pred))
        mask, pred = (mask.squeeze(), pred.squeeze())
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

    return SM / (num1), MAE/ num1
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=36, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='./data/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./CAMO/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='../pth/submit/',
                        help='the path to save model and log')
    opt = parser.parse_args()
    opt = parser.parse_args()
    

    # set the device for training 
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    cudnn.benchmark = True

    # build the model
    model = Network(channel=32).cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)
    
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    bestdice = 0
    dict_plot = {'CAMO':0 , 'test':0 }

    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer)
        # val(val_loader, model, epoch, save_path, writer)


