<h1 align="center">
<p>BEiT
</h1>

```bash
BEiT
├── model
|    ├── backbone
|    |   └──beit.py
|    └── configs
|        ├── _base_
|        |    ├── custom_dataset.py
|        |    ├── custom_runtime.py
|        |    ├── custom_schedule.py
|        |    └── upernet_beit.py
|        └── beit
|            └── custom_beit.py
└── pseudo
    ├── sub_to_json.py
    └── transforms.py
```

## 파일 설명:

`model/backbone/beit.py`: MMSegmentation에서 BEiT backbone을 사용 가능하기 위한 BEiT 구현 코드

`configs/_base_/custom_dataset.py`: Dataset 관련 custom config 파일

`configs/_base_/custom_runtime.py`: Log와 evaluation 관련 custom config 파일

`configs/_base_/custom_schedule.py`: Cosine Annealing을 적용한 schedule 관련 custom config 파일

`configs/_base_/upernet_beit.py`: Upernet 구조를 갖는 base config 파일

`configs/beit/custom_beit.py`: 위 base config 파일들을 통해 BEiT를 backbone으로 갖는 Upernet 구조의 모델 config 파일

`pseudo/sub_to_json.py`: submission 형식의 csv 파일을 json으로 변환하는 파일

`pseudo/transforms.py`: 변환 된 json 내의 object의 mask를 활용해 train data에 pasting 해주는 코드


## 모델 특징:

![swin_img](https://user-images.githubusercontent.com/69003150/140606784-e5ca9941-d23d-4eae-82bf-34409ea8174f.JPG)

Biderectional Encoder representation from image Representation (BEiT) 모델은 NLP에서 성공적으로 사용되고 있는 BERT 모델의 학습 방법에서 영감을 받아 만들어진 backbone 모델입니다. 이미지 패치를 토큰으로 간주해 토큰에 mask 씌운 것을 복원하는 것을 학습하여 image representation을 향상 시켰습니다. 


## 모델 적용 방법


### BEiT backbone 적용:

clone 받은 mmsegmentation 디렉토리 내 `mmseg/models/backbones/` 디렉토리에 `model/backbone/beit.py`를 추가해주고 `mmseg/models/backbones/__init__.py`에서 import와 \_\_all\_\_에 추가해주면 BEiT backbone 활용이 가능합니다.

### Pseudo Labeling 적용:

학습 된 BEiT 모델로 submission csv 파일을 생성 후 `pseudo/sub_to_json.py` 실행.

`mmseg/datasets/pipelines/transforms.py`에 `pseudo/transforms.py` 코드 추가하고 `mmseg/datasets/pipelines/__init__.py`에서 import 와 \_\_all\_\_에 추가하고 train_pipeline에 적용하면 cutmix를 활용 할 수 있습니다. json 변수에는 `sub_to_json.py` 코드를 통해 생성한 json 파일 디렉토리를 입력하면 됩니다.

적용 예시:
```
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512,512), multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', prob=0.001),
    dict(type='AugMix', json='/opt/ml/unilm/beit/semantic_segmentation/pseudo_dataset.json', image_root='/opt/ml/segmentation/input/data'),
    dict(type='Albu',transforms=alb_transform),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
```

## 모델 성능:

Pseudo labeling 루프를 돌때마다 성능 향상

1. without pseudo labeling: Public LB = 0.758, Private LB = 0.717
2. first pseudo labeling: Public LB = 0.762, Private LB = 0.732
3. second pseudo labeling: Public LB = 0.761, Private LB = 0.739
4. third pseudo labeling: Public LB = 0.771, Private LB = 0.751