# SwinB + OCR Dyunet3+ with CBAM

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8980b2c6-ea12-470f-a072-5d0c477c29d7/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211105%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211105T084624Z&X-Amz-Expires=86400&X-Amz-Signature=8f4b2ebf4542508a4ab2ce4ba115db5173dfa58f6ed2b2f9934d0a04cb438866&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

- dynamic head : https://arxiv.org/abs/2106.08322
- Unet3+ : https://arxiv.org/abs/2004.08790
- OCR : https://arxiv.org/abs/1909.11065
- Swin transformer : https://arxiv.org/abs/2103.14030

## Requirements

### version

```
cudatoolkit == 10.1
torchvision == 0.9.0
pytorch == 1.8.0
scikit-learn==0.22
iterative-stratification
albumentations
```

### move mmseg to your mmsegmentation/
    cp -r mmseg /to/your/mmsegmentation/


and mmsegmentation have to be installed based on your directory

    # in your mmsegmentation dir
    pip install -e .


and mmcv-full have to be installed based on required torch and torchtoolkit

    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html


## Dataset

### 1. stratified kfold

    python3 util/stratified_kfold.py

After executing this, cv_train[1-5].json and cv_val[1-5].json are created in ./stratified_kfold

### 2. create train/val annotation images and move images

for using mmsegmentation, Have to follwing mmsegmentation data structure.

this code help you to struct data structure following mmsegmentation structure.

    python3 util/convert2mmseg.py [json file] --anns_file_path_root ./stratified_kfold/

ex) cv_train1.json

    python3 util/convert2mmseg.py cv_train1 --anns_file_path_root ./stratified_kfold/

ex) cv_val1.json

    python3 util/convert2mmseg.py cv_val1 --anns_file_path_root ./stratified_kfold/


### 3. Move test image
    python3 util/convert2mmseg.py test --anns_file_path_root /opt/ml/segmentation/input/data --move_only

### 4. Move pseudo image
    cp -r /pseudo ./dataset/annotations/


**If you follwing step 1-4, Have to follwing data structure below.**

```
├──/dataset
|   ├── images
|   |   ├── cv_train1
|   |   ├── cv_train2
|   |   ├── cv_train3
|   |   ├── cv_train4
|   |   ├── cv_train5
|   |   ├── cv_val1
|   |   ├── cv_val2
|   |   ├── cv_val3
|   |   ├── cv_val4
|   |   ├── cv_val5
|   |   ├── test
|   |
|   ├── annotations
|   |   ├── cv_train1
|   |   ├── cv_train2
|   |   ├── cv_train3
|   |   ├── cv_train4
|   |   ├── cv_train5
|   |   ├── cv_val1
|   |   ├── cv_val2
|   |   ├── cv_val3
|   |   ├── cv_val4
|   |   ├── cv_val5
|   |   ├── pseudo
```

### 5. Modify data_root in \_base\_\/base_dataset.py

before modify)

    data_root = '/opt/ml/segmentation/moon/dataset/'

after modify)

    data_root = '/to/your/dataset/root'


## Create pretrain weight for mmsegmentation swinB
    # in your mmsegmentation dir
    python3 tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth pretrain/swin_base_patch4_window12_384_22k.pth

## Training
    # in your mmsegmentation dir
    python3 tools/train.py in/this/dir/model/OCR_dyunetCBAM_swinB.py

## Inference

    python3 inference.py model_cfg model_cfg_dir submission_dir

- model_cfg : model config file name
- model_cfg_dir : directory that model_cfg is saved
- submission_dir : directory that submission.csv will be saved.

ex)

    python3 inference.py OCR_dyunetCBAM_swinB.py /opt/ml/segmentation/moon/mmsegmentation/work_dirs/pseudo_OCR_dyunetCBAM_swinB_cv3 /opt/ml/segmentation/moon/submission/






