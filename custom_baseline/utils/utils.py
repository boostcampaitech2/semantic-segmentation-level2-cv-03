import random
import torch
import torch.nn.functional as F
import numpy as np
import collections
import os
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from operator import add
import json
import wandb
import cv2
from sklearn.model_selection import KFold
import logging

class_labels ={
        0: 'Background',
        1: 'General trash',
        2: 'Paper',
        3: 'Paper pack',
        4: 'Metal',
        5: 'Glass',
        6: 'Plastic',
        7: 'Styrofoam',
        8: 'Plastic bag',
        9: 'Battery',
        10: 'Clothing'
    }

cls_colors = {
    0: (0,0,0),
    1 : (192,0,128),
    2 : (0,128,192),
    3 : (0,128,64),
    4 : (128,0,0),
    5 : (64,0,128),
    6 : (64,0,192),
    7 : (192,128,64),
    8 : (192,192,128),
    9 : (64,64,128),
    10 : (128,0,192)
  }

def get_logger(name):
    logger = logging.getLogger('Segmentation')
    logger.setLevel(logging.INFO)
    logger_dir = f'./logs/'
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)
    file_handler = logging.FileHandler(os.path.join(logger_dir, name, '.log'))
    logger.addHandler(file_handler)
    return logger

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def save_model(model, version, save_type='loss'):
    save_path = os.path.join(f'./ckpts/{version}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_dir = os.path.join(save_path, f'best_{save_type}.pth')
    torch.save(model.state_dict(), save_dir)

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                        minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
    
def add_hist(hist, label_trues, label_preds, n_class):
    """
        stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist

def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def train_loop(model, loader, criterion, optimizer, scheduler, device, epoch, logger):
    model.train()
    losses = []
    hist = np.zeros((11,11))
    
    with tqdm(loader, total=len(loader), unit='batch') as tbar:
        for idx, batch in enumerate(tbar):
            tbar.set_description(f"Train Epoch {epoch+1}")
            optimizer.zero_grad()
            
            images, masks = batch['image'], batch['mask']
            images, masks = images.to(device), masks.to(device).long()
            
            preds = model(images)
            if isinstance(preds, collections.OrderedDict):
                preds = preds['out']
            elif isinstance(preds, list):
                for i in range(len(preds)):
                    pred = preds[i]
                    ph, pw = pred.size(2), pred.size(3)
                    h, w = masks.size(1), masks.size(2)
                    if ph != h or pw != w:
                        pred = F.interpolate(input=pred, size=(
                            h, w), mode='bilinear', align_corners=True)
                    preds[i] = pred
                    
            if isinstance(preds, list):
                loss = 0
                ratio = [1, 0.4, 0.2]
                for i in range(len(preds)):
                    loss += criterion(preds[i], masks) * ratio[i]
                preds = preds[0]
            else:
                loss = criterion(preds, masks)
                
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
                
            preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
            hist = add_hist(hist, masks.detach().cpu().numpy(), preds, n_class=11)
            mIoU = label_accuracy_score(hist)[2]
            losses.append(loss.item())
            
            wandb.log({
                'Learning rate': get_learning_rate(optimizer)[0],
                'Train Loss value': np.mean(losses),
                'Train mean IoU value': mIoU * 100.0,
            })
            
            if (idx+1) % (int(len(loader)//10)) == 0 or (idx+1) == len(loader):
                logger.info(f'Train Epoch {epoch+1} ==>  Batch [{str(idx+1).zfill(len(str(len(loader))))}/{len(loader)}]  |  Loss: {np.mean(losses):.5f}  |  mIoU: {mIoU:.5f}')

            tbar.set_postfix(trn_loss=np.mean(losses),
                                trn_mIoU=mIoU)
            
    return np.mean(losses), mIoU

def val_loop(model, loader, criterion, device, epoch, logger,debug=True):
    cnt=1
    model.eval()
    losses = []
    hist = np.zeros((11, 11))
    example_images = []
    
    logger.info(f"\nValid on Epoch {epoch+1}")
    with torch.no_grad():
        with tqdm(loader, total=len(loader), unit='batch') as val_bar:
            for idx, batch in enumerate(val_bar):
                val_bar.set_description(f"Valid Epoch {epoch+1}")

                images, masks = batch['image'], batch['mask']
                images, masks = images.to(device), masks.to(device).long()

                preds = model(images)
                if isinstance(preds, collections.OrderedDict):
                    preds = preds['out']
                elif isinstance(preds, list):
                    _, preds = preds
                    ph, pw = preds.size(2), preds.size(3)
                    h, w = masks.size(1), masks.size(2)
                    if ph != h or pw != w:
                        preds = F.interpolate(input=preds, size=(
                            h, w), mode='bilinear', align_corners=True)

                loss = criterion(preds, masks)
                losses.append(loss.item())

                if debug:
                    debug_path = os.path.join('.', 'debug', 'valid')
                    if not os.path.exists(debug_path):
                        os.makedirs(debug_path)

                    file_names = batch['info']
                    pred_masks = torch.argmax(preds.squeeze(), dim=1).detach().cpu().numpy()
                    for idx, file_name in enumerate(file_names):
                        pred_mask = pred_masks[idx]
                        ori_image = cv2.imread(os.path.join('/opt/ml/segmentation/input/data', file_name))
                        ori_image = ori_image.astype(np.float32)

                        for i in range(1, 11):
                            a_mask = (pred_mask == i)
                            cls_mask = np.zeros(ori_image.shape).astype(np.float32)
                            cls_mask[a_mask] = cls_colors[i]
                            ori_image[a_mask] = cv2.addWeighted(ori_image[a_mask], 0.2, cls_mask[a_mask], 0.8, gamma=0.0)

                        cv2.imwrite(os.path.join(debug_path, f"{cnt}.jpg"), ori_image)
                        cnt += 1

                input_np = cv2.imread(os.path.join('/opt/ml/segmentation/input/data', file_names[0]))
                example_images.append(wandb.Image(input_np, masks={
                    "predictions": {
                        "mask_data": preds.argmax(1)[0].detach().cpu().numpy(),
                        "class_labels": class_labels
                    },
                    "ground-truth": {
                        "mask_data": masks[0].detach().cpu().numpy(),
                        "class_labels": class_labels
                    }
                }))

                preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
                hist = add_hist(hist, masks.detach().cpu().numpy(), preds, n_class=11)
                mIoU = label_accuracy_score(hist)[2]



                if (idx + 1) % (int(len(loader) // 10)) == 0 or (idx+1) == len(loader):
                    logger.info(
                        f'Valid Epoch {epoch+1} ==>  Batch [{str(idx+1).zfill(len(str(len(loader))))}/{len(loader)}]  |  Loss: {np.mean(losses):.5f}  |  mIoU: {mIoU:.5f}')

                val_bar.set_postfix(val_loss=np.mean(losses),
                                    val_mIoU=mIoU)

            wandb.log({
                'Example Image': example_images,
                'Valid Loss value': np.mean(losses),
                'Valid mean IoU value': mIoU * 100.0,
            })

    return np.mean(losses), mIoU