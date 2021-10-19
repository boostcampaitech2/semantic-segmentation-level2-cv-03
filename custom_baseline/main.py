from matplotlib.pyplot import get
from utils.utils import *
from models.basic import *
from models.smp_models import *
from losses.ce_loss import *
from data.data import *
from utils.schedulers import *

import wandb
import logging
import argparse
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='test', help='run name')
parser.add_argument('--lr', type=float, default=1e-4, help='start learning rate')
parser.add_argument('--minlr', type=float, default=1e-3, help='minimum learning rate')
parser.add_argument('--maxlr', type=float, default=5e-3, help='maximum learning rate')
parser.add_argument('--seed', type=int, default=2021, help='minimum learning rate')
parser.add_argument('--epoch', type=int, default=40, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16)   
parser.add_argument('--num_workers', type=int, default=4)   


if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    
    config = dict(
        learning_rate = args.max_lr,
        architecture = args.name,
        seed = args.seed,
        batch_size=args.batch_size
    )
    wandb.init(config=config,project="Pstage-Seg", entity='jsw', name=args.name, save_code=True)
    
    logger = get_logger(args.name)
    
    train_loader = get_dataloader(mode='train', num_workers=args.num_workers, batch_size=args.batch_size)
    val_loader = get_dataloader(mode='val', num_workers=args.num_workers, batch_size=args.batch_size)
    
    device = get_device()
    EPOCHS = args.epoch
    model = get_fcn_r50().to(device)
    # model = UnetPlusPlus().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min =5e-3, T_max=int(EPOCHS*len(train_loader)))
    scheduler = CosineAnnealingWarmupRestarts(optimizer, len(train_loader)*EPOCHS, 1.0, args.max_lr, args.min_lr, int(len(train_loader)*EPOCHS*0.1))
    
    best_loss = float("INF")
    best_mIoU = 0
    # Train loop
    for epoch in range(EPOCHS):
        trn_loss, trn_mIoU = train_loop(model, train_loader, criterion, optimizer, scheduler, device, epoch, logger)
        val_loss, val_mIoU = val_loop(model, val_loader, criterion, device, epoch, logger)
        save_model(model, version=args.name, save_type='current')
        if best_loss > val_loss:
            logger.info(f"Best loss {best_loss:.5f} -> {val_loss:.5f}")
            best_loss = val_loss
            save_model(model, version=args.name, save_type='loss')

        if best_mIoU < val_mIoU:
            logger.info(f"Best mIoU {best_mIoU:.5f} -> {val_mIoU:.5f}")
            best_mIoU = val_mIoU
            save_model(model, version=args.name, save_type='mIoU')