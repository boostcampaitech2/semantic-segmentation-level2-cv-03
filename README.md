# AI_Tech_segmentation_competition
This repo contains the **4th solutions** on **BoostCamp AI_Tech** (2nd term) semantic segmentation competetion.  

## Contents
```
|-- README.md
`-- UperNet_swin
    |-- Inference
    |   |-- SoftVoting_Ensemble.py
    |   |-- inference.py
    |   `-- inferenceConvCRF_Ensemble.py
    |-- Preprocessing
    |   |-- EDA4kFold.ipynb
    |   `-- mmseg_mode_resampling.py
    |-- setup
    |   |-- albumentations.py
    |   `-- save_confidence.py
    |-- upernet_swinB.py
    `-- upernet_swinB_test.py
```
`upernet_swinB.py`  
: It trains model on customized dataset (ModeResampled masks)
- CV mIoU 0.728
- LB measurement of the model was not possible due to the limitation of the number of submissions
- mIoU metrics tends to be higher in LB than CV. (CV 0.699 -> LB 0.743 on default dataset)
- uses multi-scaled images [512 ~ 1024]
  

## Requirements
**Libraries**
- Ubuntu 18.04 LTS
- Python 3.7.5
- PyTorch 1.7.1
- mmcv-full 1.3.14
- mmsegmentation 0.18.0

**Hardware**
- GPU: 1 x NVIDIA Tesla V100 32G

## Train Models (GPU needed)
On a single GPU
```
python tools/train.py [path to upernet_swinB.py]
```

On multiple GPUs
```
tools/dist_train.sh [path to upernet_swinB.py] [number of GPUs]  
```  

## Datasets
**Mode Resampling**  
<img width="512" alt="image" src="https://user-images.githubusercontent.com/30382262/140639047-cd31861c-9b66-4b3b-92ee-fa25da73dbf6.png">
- find the mode in a 3x3 kernel
- fill mode value into each 9 pixels of current window  

**Mode Resampling on Masks**  
<img width="786" alt="image" src="https://user-images.githubusercontent.com/30382262/140638769-9d705d54-beb6-4aa3-9039-ab03c875ec92.png">
<img width="786" alt="image" src="https://user-images.githubusercontent.com/30382262/140638815-d520b906-36d3-439a-8258-fc0de7db95f6.png">
<img width="786" alt="image" src="https://user-images.githubusercontent.com/30382262/140638818-69051ef6-44f0-42f2-a767-89b0d2465a88.png">  
[네이버 커넥트재단 - 재활용 쓰레기 데이터셋 / CC BY 2.0]


## Performance in Customized model & dataset
<img width="800" alt="image" src="https://user-images.githubusercontent.com/30382262/140639399-0be529f5-ae6b-4262-9531-4ea37833797e.png">
- Noticeable performance improvement was found by using ModeResampled masks (blue line).
- LB.725 → LB.738
