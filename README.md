# Overview

## 2D pose estimation
- 아래 그림처럼 2d pose estimation을 위한 keypoint를 표현하는 방법이 여러가지 입니다. 이번 대회에서는 SMPL body joint를 예측하는 문제 입니다.
![](https://github.com/hongsukchoi/Pose2Mesh_RELEASE/blob/master/asset/joint_sets.png)

## 3D pose estimation from image
- 3D pose estimation이 어려운 이유를 먼저 알고 넘어갈 필요성이 있습니다. 3D pose estimation은 카메라에 찍힌 2D image로 부터 계산 되는데 **실제 세상에서 찍힌 물체가 사진에 담기는 과정에서 소실된 정보가 무엇인지 알아야합니다.**

### Depth information
- 먼저 아래는 남자가 스쿼트 자세를 취하고 있는 사진입니다. **사람은 이 사진을 보면 한쪽팔이 비록 안보이더라도 양팔 모두를 앞으로 뻗고 있는 것을 알 수 있고 오른쪽 무릎이 왼쪽 무릎보다 더 먼거리에 있다는 것을 알고 있습니다.** 하지만 2D image만 가지고는 해당 정보를 알 수가 없습니다. 소실된 카메라와 물체간 거리정보를 복원 시켜야 하는 문제가 있습니다.
  - ![](https://health.chosun.com/site/data/img_dir/2021/05/20/2021052000854_0.jpg) 

- 아래 처럼 사진에 담긴 물체의 크기가 다르지만 카메라와 같은 거리에 있을 수 있고, 물체의 크기가 같지만 카메라와 서로 다른 거리에 있을 수 있습니다. 
  - ![](https://images.deepai.org/converted-papers/1907.11346/x4.png)

- Depth informaition은 크게 카메라와 물체간의 거리, 그리고 물체의 상대적 거리 (오른쪽 무릎이 왼쪽 무릎뒤에 있는 것 같은 상대적 정보) 2가지가 있다고 볼 수 있겠습니다.

### Rotation information
- 아래 사진은 여러대의 카메라가 서로 다른 위치에서 같은 포즈를 취하고 있는 사람을 찍은 것입니다. 3d pose estimation은 이처럼 서로 다른 각도에서 찍은 사진이더라도 3d 좌표(world coordinate)에 동일하게 표현되어야 합니다. 하지만 2D image만 보면 **물체가 x,y,z 축으로 얼마나 틀어져있는지 회전 정보를 알 수가 없습니다.** 이 또한 복원 시켜야 하는 문제가 있습니다. 
  - ![](https://i.ibb.co/g3wygTq/image14.png)

## Solution
### 주어진 정보
- 원래 3D pose estimation은 2d image만 주어지고 소실된 Depth 와 Rotation 정보를 복원한 3D 좌표를 예측하는 task 입니다. 
- 하지만 **이번 대회에서는 camera의 intrinsic, extrinsic 정보가 주어졌다는 것**이 가장 큰 차이점입니다.
- camera intrinsic, extrinsic 정보에 대해 간략히 설명하면 카메라에 이미지가 저장되는 원리를 알아야합니다. 아래 그림과 같이 실제 존재하는 나무를 카메라로 찍으면 카메라 내부 이미지 센서에 pixel의 형태로 저장됩니다. 이때 **실제 물체와 카메라 간의 거리, 그리고 카메라의 x,y,z 축의 틀어짐 정도와 같은 카메라 외부 정보를 담고 있는것이 extrinsic 정보입니다. 그리고 나무는 카메라 내부에서 pixel의 형태로 저장이 되어야 하는데 실제 카메라에 들어온 물체와 이미지로 저장될때 렌즈와 이미지센서간 초점 거리라던지 이미지 센서가 얼마나 틀어져있는지와 같은 카메라 내부정보를 담고 있는것이 intrinsic 정보**입니다. 
  - ![](https://www.mathworks.com/help/vision/ug/calibration_cameramodel_coords.png)

### 문제를 다시 생각해보기
- camera의 intrinsic, extrinsic 정보를 안다는것은 물체와 카메라가 상대적으로 얼마나 떨어져있는지, 물체를 찍을때 카메라의 x,y,z 축이 얼마나 틀어져있는지 알 수 있습니다.
- 하지만 물체와 카메라간의 거리는 알 수 있어도 세부적으로 목이 더 앞으로 나왔다거나, 아직 오른팔이 왼팔보다 뒤에 위치하고 있다는 상대적 거리에 대한 정보는 아직 알 수 없습니다.   
- 그럼 2d image의 keypoint 좌표를 알고 각 keypoint 별 상대적 거리만 복원시킨다면 나머지는 camera intrinsic, extrinsic 정보를 통해서 3d pose estimation을 할 수 있게 됩니다.

### pretrain model
- Object detection (사람 영역 추출) + SMPL body joint (keypoint 좌표 추출) + 3d 좌표 계산을 할 수 있는 pretrained model이 무엇이 있을지 집중해서 찾았습니다.
- 최종적으로 우리가 썼던 [metrabs](https://github.com/isarandi/metrabs)는 이를 제공하는 모델임을 확인하였고 이를 활용해서 학습을 하기로 결정했습니다.
- 사용한 모델은 대회측에서 제공한 base model인 posenet을 그대로 사용하였습니다. 
- 기본적으로 3d pose estimation에서 소실된 거리와 회전 정보를 처음부터 복원을 했다면 결과가 좋지 않았을 텐데, 데이터 분석을 통해 적합한 pretrained model 선정 및 예측 정보의 간소화를 통해 예측범위를 축소한 것이 이번 문제 해결의 키포인트 였습니다.
   
# HOW TO Train and Predict

```
$ run.sh
```

# code file description

```
/RESULT
├── README.md
├── config
│   └── train_config.yml
├── data -> /DATA/Final_DATA/
├── extract_train_info.py 
├── model
│   ├── model.py 
├── modules
│   ├── dataset.py
│   ├── earlystoppers.py
│   ├── metrics.py
│   ├── pose_utils.py
│   ├── recorders.py
│   ├── trainer.py
│   ├── utils.py
│   └── vis.py
├── pre_models
│   ├── metrabs_multiperson_smpl
│   ├── metrabs_multiperson_smpl_combined
│   └── metrabs_singleperson_smpl
├── results
│   └── test
│   └── train
├── run.sh
├── sample_submission.json
├── pred_submission.json
├── submission.sh
├── test.py
└── train.py
```

## config
  - train_config.yml : Training and test configuration 
## data
  - Data path 
## model
  - model.py : posenet(restnet 18 backbone + resnet18 HeadNet + MLP layer for pose embedding)
## modules
  - dataset.py : make custom dataset for train, validation and test
## pre_models
  - metrabs : pretrained model that we requested 
## extract_train_info.py
  - extract training meta data
## train.py
  - training code
## test.py
  - test code
## pred_submission.json
  - predict result from test.py

# output description

```
├── results
│   └── test
│   └── train
│       └── POSENET_20210702084337 # example
│           └── best_xx.pt
│           └── loss.png
│           └── model.pt
│           └── record.csv
│           └── score.jpg
│           └── train_config.yaml
│           └── train_log.log

```
- best_xx.pt : saved model each epoch
- loss.png : loss curve
- record.csv and train_log.log : information of model and loss score
- train_config.yaml : training configuration   


