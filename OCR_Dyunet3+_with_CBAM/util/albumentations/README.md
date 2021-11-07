# Albumentation

## Usage
 1. /opt/ml/segmentation/moon/mmsegmentation/mmseg/datasets/pipelines 내부에 붙여넣기

or 

 2. transforms.py 내부에 MyAlbu class를 자신의 /opt/ml/segmentation/moon/mmsegmentation/mmseg/datasets/pipelines/transforms.py에 추가한 뒤, /opt/ml/segmentation/moon/mmsegmentation/mmseg/datasets/pipelines/__init__.py에서 transforms import 시 MyAlbu class 추가 import