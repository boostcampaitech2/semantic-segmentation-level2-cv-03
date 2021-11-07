# semantic-segmentation-level2-cv-03
semantic-segmentation-level2-cv-03 created by GitHub Classroom

cutmix.py 사용법

## Path

background_img : background가 될 이미지들을 모아놓은 폴더
dataset_base : cutmix에 사용될 이미지들이 모여있는 폴더

```python
background_img = os.listdir('/opt/ml/segmentation/moon/background')
dataset_base = 'input/data'
```
## area, res 조건으로 cutmix설정
- res가 2일떄 하나의 이어져있는 물체
- area는 해당 mask가 차지하는 넓이

```python
res, _ ,a,center = cv2.connectedComponentsWithStats(mask)    
            # object가 하나로 이어져 있지 않거나 크기가 너무 작으면 패스
            if res>2 or area <25000 or area>1700000:
                idx+=1
                continue
```