# 0. Overview
UperNet_swin (https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-03/tree/main/UperNet_swin)  
위의 디렉토리에서 박승찬의 작업물을 더 자세히 확인하실 수 있습니다.  

</br>


# 1. Introduction  

</br>

<p align="center">
   <img src="https://kr.object.ncloudstorage.com/resume/boostcamp/boostcamplogo.png"/>
</p>
<p align="center">
   <img src="https://kr.object.ncloudstorage.com/resume/boostcamp/boostcamplogo2.png"/>
</p>

본 과정은 NAVER Connect 재단 주관으로 인공지능과 딥러닝 Production의 End-to-End를 명확히 학습하고 실무에서 구현할 수 있도록 훈련하는 약 5개월간의 교육과정입니다. 전체 과정은 이론과정(U-stage, 5주)와 실무기반 프로젝트(P-stage, 15주)로 구성되어 있으며, 세 번째 대회인 `Semantic Segmatation`과제에 대한 **Level2 - 03조** 의 문제해결방법을 기록합니다.
  
<br/>

## 🧙‍♀️ Dobbyision - 도비도비전잘한다  
”도비도 비전을 잘합니다”  
### 🔅 Members  

김지수|박승찬|박준수|배지연|이승현|임문경|장석우
:-:|:-:|:-:|:-:|:-:|:-:|:-:
![image1][image1]|![image2][image2]|![image3][image3]|![image4][image4]|![image5][image5]|![image6][image6]|![image7][image7]
[Github](https://github.com/memesoo99)|[Github](https://github.com/ark10806)|[Github](https://github.com/JJONSOO)|[Github](https://github.com/jiiyeon)|[Github](https://github.com/lsh3163)|[Github](https://github.com/larcane97)|[Github](https://github.com/sw-jang)


### 🔅 Contribution  
- `김지수` &nbsp; Modeling • Data Augmentation  
- `박승찬` &nbsp; Modeling • Cross-Validation   
- `박준수` &nbsp; Modeling • Data Augmentation  
- `배지연` &nbsp; Modeling • EDA  
- `이승현` &nbsp; Modeling • Cross-Validation
- `임문경` &nbsp; Modeling • Model Augmentation  
- `장석우` &nbsp; Modeling • Pseudo Labeling 

[image1]: https://kr.object.ncloudstorage.com/resume/boostcamp/00.png
[image2]: https://kr.object.ncloudstorage.com/resume/boostcamp/01.png
[image3]: https://kr.object.ncloudstorage.com/resume/boostcamp/02.png
[image4]: https://kr.object.ncloudstorage.com/resume/boostcamp/03.png
[image5]: https://kr.object.ncloudstorage.com/resume/boostcamp/04.png
[image6]: https://kr.object.ncloudstorage.com/resume/boostcamp/05.png
[image7]: https://kr.object.ncloudstorage.com/resume/boostcamp/06.png


<br/>

# 2. Project Outline  

![competition_title](https://user-images.githubusercontent.com/68527727/140634092-9504bb59-3058-443b-b93f-538b5117cbe0.png)

- Task : Semantic Segmantation
- Date : 2021.10.18 - 2021.11.4 (3 weeks)
- Description : 쓰레기 사진을 입력받아서 `일반 쓰레기, 플라스틱, 종이, 유리 등`를 추측하여 `10개의 class`에 해당하는 객체의 영역을 구합니다.   
- Image Resolution : (512 x 512)
- Train : 3,272
- Test : 819

</br>

### 🏆 Final Score  
<p align="center">
   <img src="https://user-images.githubusercontent.com/68527727/140634712-aeb9b875-b37a-4957-a273-ad019def2b2a.png">
</p>

<br/>

# 3. Solution
![process](https://user-images.githubusercontent.com/68527727/140636725-b676645d-b106-4078-b64f-85aad4d6ee7d.png)

### KEY POINT

- mmsegmantation을 활용하여 실험 환경을 구축합니다.  
- 모델이 학습하기 어려워하는 클래스의 예측률을 높일 수 있는 조건을 찾아갑니다.  
- 객체 안의 작은 점들이 노이즈로 작동하지 않도록 전처리 했습니다. 
- Loss, Augmentation, lr scheduler 등을 하나씩 바꾸고 각 기능의 효과를 분석하여 최적의 조건을 찾아갔습니다.    

&nbsp; &nbsp; → 검증된 모델의 실험을 기반으로 backbone을 다양하게 확보하여 앙상블을 통한 일반화 성능 향상을 추구합니다.  

[process]: https://kr.object.ncloudstorage.com/resume/boostcamp/pipeline.png

### Checklist  
- [x] Data Curation
- [x] Test Time Augmentation
- [x] Loss - Cross entropy
- [x] Ensemble(BEiT, UperSwinB, OCR+DyUnetCBAM3)
- [x] Background patches
- [x] Oversampling for class imbalance problem  
- [x] Pseudo labeling
- [x] Cutmix  
- [x] Straified k-fold
- [x] Change backbone model  
- [x] Post Processing


### Evaluation

| Method| single| K-fold| Pseudo Labeling| Total|
| --- | :-: | :-: | :-: | :-: |
|BEiT| 0.758|-|0.771|-|
|UperSwinB|0.738|0.743|-|-|
|OCR+DyUnetCBAM3|0.758|0.768|-|0.769|
|Ensemble|-|-|-|0.775|

</br>

# 4. How to Use


```
.
├──/dataset
|   ├── train.json
|   ├── test.json
|   ├── /train
|   ├── /test
├──/semantic-segmentation-level2-cv-03
│   ├── model1
│         ├── config.py
│         └── readme.md
│   ├── model2
│         ├── config.py
│         └── readme.md
```

- `model`안에는 각각 **config.py** •  **readme.md**가 들어있습니다  
- 사용자는 전체 코드를 내려받은 후 설명서에 따라 옵션을 지정하여 개별 라이브러리의 모델을 활용할 수 있습니다
- 각 라이브러리의 구성요소는 `readme.md`에서 확인할 수 있습니다  
