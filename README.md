# Experiment
재활용 품목 분류를 위한 Semantic Segmantation Competition을 위해 실험한 내용입니다.  

- **Date** : 2021.10.18 - 2021.11.12  
- **Outline** : 
    -  Input : 쓰레기 객체가 담긴 이미지. segmentation annotation은 COCO format으로 주어집니다.
    -  Category : 사진에는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 객체가 포함되어있습니다.  
    -  Mask : 모델은 pixel 좌표에 따라 카테고리 값을 리턴합니다.
    -  Metric : mIoU(Mean Intersection over Union)  
    -  Output : submission 양식에 맞게 csv 파일을 만들어 제출합니다.


### Structure
```

.
├── EDA
|     ├── EDA.py
|
├── Train_Experiment
|     ├── data
|           ├── data_loader.py
|     ├── model
|           ├── loss.py
|           ├── model.py
|     ├── trainer
|           ├── __init__.py
|           ├── trainer.py
|     ├── util
|           ├── __init__.py
|           ├── util.py
|     ├── train.py
|     ├── parse_config.py
|     ├── config.json

```
`EDA.py` : 주어진 이미지의 annotation을 활용하여 mask의 위치분포를 파악합니다.  
`train.py` : config.json의 정보를 토대로 모델을 학습합니다.  
`config.json` : 모델 학습 시 필요한 세부 정보(batch_size, optimizer 등)를 관리합니다.  


### How to Use
```python

# basic
$ python train.py -c {config.json} -m "experiment"

# train all data
$ python train.py -c {config.json} -m "all"

# if you want to chance just lr(learning rate)
$ python train.py -c {config.json} -lr "value" -m "experiment"

# if you want to chance just batch_size
$ python train.py -c {config.json} -bs "value" -m "experiment"

```
