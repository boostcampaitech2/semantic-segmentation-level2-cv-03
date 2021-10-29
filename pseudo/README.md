<h1 align="center">
<p>Pseudo Labeling Tools
</h1>

`sub_to_json.py`: submission 형식의 csv를 json으로 변환
    
    수정이 필요한 변수들:
    fold_path: train.json
    pesudo: 제출 csv
    best_submission: pseudo label을 생성할 csv [(256,256) Resize 하지 않은 csv]


`transforms.py`: 변환 된 json을 통해 CutMix

    1. mmseg/datasets/pipelines/transforms.py에 코드 추가
    2. mmseg/datasets/pipelines/__init__.py에서 import와 __all__에 추가
    3. 이후 train_pipeline에 적용
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