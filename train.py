""" 학습 코드

TODO:

NOTES:

REFERENCE:
    * MNC 코드 템플릿 train.py

UPDATED:
"""

import os
import random
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import numpy as np
import torch.optim as optim
import madgrad
from torch.utils.data import DataLoader
from modules.metrics import get_metric_fn
from modules.dataset import CustomDataset
from modules.trainer import Trainer
from modules.utils import load_yaml, save_yaml, get_logger, make_directory
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder
import torch
import torch.nn as nn
from model.model import get_pose_net
import pdb
torch.multiprocessing.set_sharing_strategy('file_system')
# DEBUG
DEBUG = False

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yml')
config = load_yaml(TRAIN_CONFIG_PATH)
# SEED
RANDOM_SEED = config['SEED']['random_seed']

# DATALOADER
NUM_WORKERS = config['DATALOADER']['num_workers']
PIN_MEMORY = config['DATALOADER']['pin_memory']

# TRAIN
EPOCHS = config['TRAIN']['num_epochs']
BATCH_SIZE = config['TRAIN']['batch_size']
LEARNING_RATE = config['TRAIN']['learning_rate']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']
MODEL = config['TRAIN']['model']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
LOSS_FN = config['TRAIN']['loss_function']
METRIC_FN = config['TRAIN']['metric_function']
INPUT_SHAPE = config['TRAIN']['input_shape']
OUTPUT_SHAPE = config['TRAIN']['output_shape']
OUTPUT_DEPTH = config['TRAIN']['output_depth']
RESNET_TYPE = config['TRAIN']['resnet_type']
MODEL_TYPE = config['TRAIN']['model_type']
PATCH_SIZE = config['TRAIN']['patch_size']
EMBED_DIM= config['TRAIN']['embed_dim']
DEPTH = config['TRAIN']['depth']
NUM_HEADS = config['TRAIN']['num_heads']
MLP_RATIO = config['TRAIN']['mlp_ratio']
COORD = config['TRAIN']['coord']

# TRAIN SERIAL
KST = timezone(timedelta(hours=9))
TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
TRAIN_SERIAL = f'{MODEL}_{TRAIN_TIMESTAMP}' if DEBUG is not True else 'DEBUG'

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', 'train', TRAIN_SERIAL)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']


if __name__ == '__main__':

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set train result directory
    make_directory(PERFORMANCE_RECORD_DIR)

    # Set system logger
    system_logger = get_logger(name='train',
                               file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'train_log.log'))
    # Load dataset & dataloader
    print('loading meta data..')
    meta_data = torch.load(os.path.join(DATA_DIR, 'train_meta.pt'))

    meta_data.sort(key=lambda x: x.get('mpjpe_x'))
    total_meta_data = len(meta_data)
    print(total_meta_data)
    meta_data = meta_data[:int(total_meta_data*0.95)]

    meta_data.sort(key=lambda x: x.get('mpjpe_y'))
    total_meta_data = len(meta_data)
    print(total_meta_data)
    meta_data = meta_data[:int(total_meta_data*0.95)]

    depth_max = np.max(np.asarray([np.max(np.abs(d['scale'] - d['extrinsic'][-1,-1]),axis=0) for d in meta_data]),axis=0) # [x, y ,z]
    random.shuffle(meta_data)
    total_train_set = len(meta_data)
    num_train_set = int(total_train_set * 0.9)
    train_db = meta_data[:num_train_set]
    val_db = meta_data[num_train_set:]

    train_dataset = CustomDataset(data_dir=DATA_DIR, mode='train', input_shape=INPUT_SHAPE, output_depth=OUTPUT_DEPTH,
                                  db=train_db, model_type=MODEL_TYPE, depth_max=depth_max, coord=COORD)
    validation_dataset = CustomDataset(data_dir=DATA_DIR, mode='val', input_shape=INPUT_SHAPE, output_depth=OUTPUT_DEPTH,
                                       db=val_db, model_type=MODEL_TYPE, depth_max=depth_max, coord=COORD)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    
    print('Train set samples:',len(train_dataset),  'Val set samples:', len(validation_dataloader))
    print('depth_max:',depth_max)

    # Load Model
    joint_num = 24
    model = get_pose_net(RESNET_TYPE, OUTPUT_DEPTH, OUTPUT_SHAPE, is_train=True, joint_num=joint_num).to(device)

    # Set optimizer, scheduler, loss function, metric function
    optimizer = madgrad.MADGRAD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7*len(train_dataloader), gamma=0.3)
    metric_fn = get_metric_fn

    # Set trainer
    trainer = Trainer(model, device, metric_fn, depth_max, optimizer, scheduler, logger=system_logger)

    # Set earlystopper
    early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE, verbose=True, logger=system_logger)

    # Set performance recorder
    key_column_value_list = [
        TRAIN_SERIAL,
        TRAIN_TIMESTAMP,
        MODEL,
        OPTIMIZER,
        LOSS_FN,
        METRIC_FN,
        EARLY_STOPPING_PATIENCE,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        RANDOM_SEED]

    performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=system_logger,
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=scheduler)

    # Train
    save_yaml(os.path.join(PERFORMANCE_RECORD_DIR, 'train_config.yaml'), config)
    criterion = 1E+8
    for epoch_index in tqdm(range(EPOCHS)):

        trainer.train_epoch(train_dataloader, epoch_index, MODEL_TYPE, INPUT_SHAPE, OUTPUT_DEPTH)
        trainer.validate_epoch(validation_dataloader, epoch_index, MODEL_TYPE, INPUT_SHAPE, OUTPUT_DEPTH)

        # Performance record - csv & save elapsed_time
        performance_recorder.add_row(epoch_index=epoch_index,
                                     train_loss=trainer.train_mean_loss,
                                     validation_loss=trainer.val_mean_loss,
                                     train_score=trainer.train_score,
                                     validation_score=trainer.validation_score)
        
        # Performance record - plot
        performance_recorder.save_performance_plot(final_epoch=epoch_index)
        """
        # early_stopping check
        early_stopper.check_early_stopping(loss=trainer.val_mean_loss)

        if early_stopper.stop:
            print('Early stopped')
            break
        """
        if trainer.train_mean_loss < criterion:
            criterion = trainer.train_mean_loss
            performance_recorder.weight_path = os.path.join(PERFORMANCE_RECORD_DIR, 'best_{:02d}.pt'.format(epoch_index))
            performance_recorder.save_weight()
        

