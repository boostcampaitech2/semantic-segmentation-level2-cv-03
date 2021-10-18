from matplotlib.pyplot import get
from utils.utils import *
from models.basic import *
from models.smp_models import *
from losses.ce_loss import *
from data.data import *

import wandb
import logging
import torch.nn as nn

if __name__ == '__main__':
    seed = 2021
    set_seed(seed)
    device = get_device()
    
    wandb.init(config= {},project="Pstage-Seg", entity='jsw')
    
    logger = logging.getLogger('Segmentation')
    logger.setLevel(logging.INFO)
    logger_dir = f'./logs/'
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)
    file_handler = logging.FileHandler(os.path.join(logger_dir, '2.log'))
    logger.addHandler(file_handler)
    
    train_loader = get_dataloader(mode='train', num_workers=4, batch_size=16)
    val_loader = get_dataloader(mode='val', num_workers=4, batch_size=16)
    
    EPOCHS = 60
    # model = get_fcn_r50().to(device)
    model = UnetPlusPlus().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS, anneal_strategy='cos')
    
    best_loss = float("INF")
    best_mIoU = 0
    for epoch in range(EPOCHS):
        trn_loss, trn_mIoU = train_loop(model, train_loader, criterion, optimizer, scheduler, device, epoch, logger)
        val_loss, val_mIoU = val_loop(model, val_loader, criterion, device, epoch, logger)
        save_model(model, version=2, save_type='current')
        if best_loss > val_loss:
            logger.info(f"Best loss {best_loss:.5f} -> {val_loss:.5f}")
            best_loss = val_loss
            save_model(model, version=2, save_type='loss')

        if best_mIoU < val_mIoU:
            logger.info(f"Best mIoU {best_mIoU:.5f} -> {val_mIoU:.5f}")
            best_mIoU = val_mIoU
            save_model(model, version=2, save_type='mIoU')