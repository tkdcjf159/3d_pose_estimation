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


