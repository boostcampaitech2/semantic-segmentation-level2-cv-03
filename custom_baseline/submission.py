import albumentations as A
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import multiprocessing as mp

from models.basic import *
from utils.utils import *
from data.data import *

def dense_crf_wrapper(args):
    return dense_crf(args[0], args[1])

def dense_crf(img, output_probs):
    MAX_ITER = 50
    POS_W = 3
    POS_XY_STD = 3
    Bi_W = 4
    Bi_XY_STD = 49
    Bi_RGB_STD = 5

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


def test_crf(model, test_loader, device):
    size = 256
    tar_size = 512
    resize = A.Resize(size, size)
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    torch.multiprocessing.set_start_method('spawn', force=True)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            imgs = batch['image']
            file_names = batch['info']
            # inference (512 x 512)
            outs = model(imgs.to(device))['out']
            
            if isinstance(outs, list):
                outs = outs[0]
                ph, pw = outs.size(2), outs.size(3)
                if ph != tar_size or pw != tar_size:
                    outs = F.interpolate(input=outs, size=(
                        tar_size, tar_size), mode='bilinear', align_corners=True)
            
            probs = F.softmax(outs, dim=1).detach().cpu().numpy()
            
            pool = mp.Pool(mp.cpu_count())
            
            images = imgs.detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
            if images.shape[1] != tar_size or images.shape[2] != tar_size:
                images = np.stack([resize(image=im)['image'] for im in images], axis=0)
            probs = np.array(pool.map(dense_crf_wrapper, zip(images, probs)))
            pool.close()
            # probs = np.array(dense_crf(images, probs))
            
            oms = np.argmax(probs.squeeze(), axis=1)
            
            # resize (256 x 256)
            temp_mask = []
            temp_images = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            for img, mask in zip(temp_images, oms):
                if mask.shape[0] != 256 or mask.shape[1] != 256:
                    transformed = resize(image=img, mask=mask)
                    mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([file_name for file_name in file_names])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array

def test(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    resize = A.Resize(size, size)

    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            imgs = batch['image']
            file_names = batch['info']
            
            # inference (512 x 512)
            outs = model(imgs.to(device))['out']
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            temp_images = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            for img, mask in zip(temp_images, oms):
                if mask.shape[0] != 256 or mask.shape[1] != 256:
                    transformed = resize(image=img, mask=mask)
                    mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([file_name for file_name in file_names])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


def make_submission(file_names, preds):
    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/sample_submission.csv', index_col=None)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv("./baseline_crf.csv", index=False)
    
if __name__ == '__main__':
    device = get_device()
    model = get_fcn_r50().to(device)
    model.load_state_dict(torch.load(os.path.join('/opt/ml/segmentation/custom_baseline/ckpts/0', 'best_mIoU.pth')))
    loader = get_dataloader(mode='test', num_workers=3, batch_size=32)
    file_names, preds_array = test_crf(model, loader, device)
    make_submission(file_names, preds_array)