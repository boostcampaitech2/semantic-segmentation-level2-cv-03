import json

dumm = {}

dumm['name'] = 'DeepLabV3+ConvMixer'
dumm['n_gpu'] = 1
dumm['path']['dataset'] = '../input/data/'
dumm['path']['data_description'] = '../input/data/'
dumm['path']['pretrainedweight'] = ''
dumm['path']['save_checkpoint']['dir'] = './saved/' + 'DeepLabV3_ConvMixer_best_model.pt'
dumm['path']['save_checkpoint']['file_name'] = 'DeepLabV3_ConvMixer_best_model.pt'
dumm['path']['inference']['sample'] = './' + 'sample_submission.csv'
dumm['path']['inference']['submission'] = './submission/' + 'deeplabv3_ConvMixer_best_model.csv'

dumm['dataloader']['type'] = 'CustomDataLoader'
dumm['dataloader']['args']['batch_size'] = 4
dumm['dataloader']['args']['num_workers'] = 2

dumm['arch']['type'] = 'DeepLabV3'
dumm['arch']['type']['args'] = {}
dumm['loss']['type'] = 'CrossEntropyLoss'
dumm['optimizer']['type']= 'AdamW'
dumm['optimizer']['args']['lr']= 0.01
dumm['optimizer']['args']['weight_decay']= 1e-6
dumm['lr_scheduler']['type'] = 'CosineAnnealingLR'
dumm['num_epoch'] = 100

dumm['wandb']['project'] = 'Pstage_seg_model_extra'
dumm['wandb']['entity'] = 'boostcampaitech2-object-detection-level2-cv-03'

json_dumm = json.dumps(dumm)
file_path = './config.json'

with open(file_path, 'w') as outfile:
    json.dump(json_dumm, outfile)