import torch
import random
import numpy as np
import argparse
import collections
import wandb
import data.data_loaders as module_data
import model.loss as module_loss
import model.model as module_arch
import trainer as module_trainer
from parse_config import ConfigParser
from util import prepare_device


def main(config, train_mode):

    # seed 고정
    random_seed = 21
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # data_path
    dataset_path = config['path']['dataset']
    decription_path = config['path']['data_description']
    train_path = decription_path + '/train.json'
    train_all_path = decription_path + '/train_all.json'
    val_path = decription_path + '/val.json'

    # setup data_loader instances
    batch_size = config['dataloader']['args']['batch_size']
    num_workers = config['dataloader']['args']['num_workers']
    data_loader = config.init_obj('data_loader', module_data)

    if train_mode == 'experiment':
        train_dataset = data_loader(data_dir=train_path, dataset_path = dataset_path ,mode='train')
        val_dataset = data_loader(data_dir=val_path, dataset_path = dataset_path, mode='train')
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers,
                                                    collate_fn=module_data.collate_fn)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    collate_fn=module_data.collate_fn)
    else :
        train_dataset = data_loader(data_dir=train_all_path, dataset_path = dataset_path, mode='train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    collate_fn=module_data.collate_fn)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)

    # prepare for (multi-device) GPU training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # train
    pretrained_weight_path = config['path']['pretrainedweight']
    if pretrained_weight_path:
        model.load_state_dict(torch.load(pretrained_weight_path), strict=False)
    
    N = config['num_epoch']
    saved_dir = config['path']['save_checkpoint']['dir']
    file_name = config['path']['save_checkpoint']['file_name']
    if train_mode == 'experiment':
        train = module_trainer.experiment_trainer(num_epochs = N, model = model,
                                                    train_loader = train_loader, val_loader = val_loader,
                                                    criterion = criterion, optimizer = optimizer,
                                                    saved_dir = saved_dir, file_name = file_name,
                                                    device = device)
    else:
        train = module_trainer.all_trainer(num_epochs = N, model = model,
                                            train_loader = train_loader,
                                            criterion = criterion, optimizer = optimizer,
                                            saved_dir = saved_dir, file_name = file_name,
                                            device = device)
    
    wandb.init(project=config['project'], entity=config['entity'])
    wandb.watch(model)
    train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    
    # choose train mode
    args.add_argument('-m', '--mode', default='experiment', type=str,
                      help='choose train mode from two types(experiment/all)')
    train_mode = args.pop()
    
    main(config, train_mode)
