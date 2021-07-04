""" 학습 코드

TODO:

NOTES:

REFERENCE:
    * MNC 코드 템플릿 test.py

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
import pandas as pd
import glob
from collections import defaultdict
import json
import tensorflow as tf
torch.multiprocessing.set_sharing_strategy('file_system')
# DEBUG
DEBUG = False

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yml')
config = load_yaml(CONFIG_PATH)
# SEED
RANDOM_SEED = config['SEED']['random_seed']

# DATALOADER
NUM_WORKERS = config['DATALOADER']['num_workers']
PIN_MEMORY = config['DATALOADER']['pin_memory']

# TEST
EPOCHS = config['TEST']['num_epochs']
BATCH_SIZE = config['TEST']['batch_size']
MODEL = config['TEST']['model']
METRIC_FN = config['TEST']['metric_function']
INPUT_SHAPE = config['TEST']['input_shape']
OUTPUT_SHAPE = config['TEST']['output_shape']
OUTPUT_DEPTH = config['TEST']['output_depth']
RESNET_TYPE = config['TEST']['resnet_type']
MODEL_TYPE = config['TEST']['model_type']
PATCH_SIZE = config['TEST']['patch_size']
EMBED_DIM= config['TEST']['embed_dim']
DEPTH = config['TEST']['depth']
NUM_HEADS = config['TEST']['num_heads']
MLP_RATIO = config['TEST']['mlp_ratio']
COORD = config['TEST']['coord']

# TEST SERIAL
KST = timezone(timedelta(hours=9))
TEST_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
TEST_SERIAL = f'{MODEL}_{TEST_TIMESTAMP}' if DEBUG is not True else 'DEBUG'

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', 'test', TEST_SERIAL)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']


def cam2pixel(cam_coord, f, c, IMG_SCALE):
    u = (cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8)) * f[0] + c[0]
    v = (cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8)) * f[1] + c[1]
    s = cam_coord[:, 2]
    img_coord = np.concatenate((u[:,None] * IMG_SCALE, v[:,None] * IMG_SCALE, s[:,None]),1)
    uv = np.concatenate((u[:,None] , v[:,None] ),1)
    return img_coord, uv, s

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def pixel2cam(suv1, intrinsic):
    inv_intrinsic = np.linalg.inv(intrinsic)
    cam_coord = (inv_intrinsic @ suv1.transpose(1,0)).transpose()
    return cam_coord

def cam2world(cam_coord, rotation, offset):
    inv_rotation = np.linalg.inv(rotation)
    world_coord = (inv_rotation @ cam_coord.transpose(1,0)).transpose() - (inv_rotation @ offset)
    return world_coord

def make_test_meta():
    ############################################### GET META DATA ################################################
    TEST_DATA_PATH = '/DATA/Final_DATA/task04_test'
    test_image_list = glob.glob(TEST_DATA_PATH+'/images/*/*.jpg')
    test_camera_list = glob.glob(TEST_DATA_PATH+'/camera/*.json')

    image_dict = defaultdict()
    for test_image_path in test_image_list:
        test_args = os.path.basename(test_image_path).replace('.jpg','').split('_')
        pose_type = test_args[0]
        person_type = test_args[1]
        camera_type = test_args[2]
        pose_sequence = '{0:04d}'.format(int(test_args[3]))
        image_key = '{}_{}_{}_{}'.format(pose_type, person_type, camera_type, pose_sequence)
        image_dict[image_key] = test_image_path
    
    camera_dict = defaultdict()
    for test_camera_path in test_camera_list:
        camera_info = open(test_camera_path, 'r')
        camera_json = json.load(camera_info)
        camera_dict[os.path.basename(test_camera_path).split('.json')[0]] = camera_json

    skeleton = ((0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
                     #(9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22),
                     (12, 13), (12, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22),
                     (21, 23))
    joint_edges = np.asarray(skeleton)
    base_dir = '/DATA/Final_DATA'
    CROP_X = 440
    CROP_Y = 200
    IMG_SCALE = 1920
    print('total test image : {}'.format(len(test_image_list)))
    ############################################################################################################


    ############################################### GET METRABS ################################################
    metrabs_model_path = './pre_models'
    model_combined_type = 'metrabs_multiperson_smpl_combined'
    model_single_type = 'metrabs_multiperson_smpl'
    metrabs_combined_model = tf.saved_model.load(os.path.join(metrabs_model_path, model_combined_type))
    metrabs_single_model = tf.saved_model.load(os.path.join(metrabs_model_path, model_single_type))
    ############################################################################################################

    ############################################### GET TEST METADATA ################################################
    pbar = tqdm(image_dict.items())
    metrabs_db = []
    for image_key, test_image_path in pbar:   
        pbar.set_description(test_image_path)
   
        image_dir = test_image_path.split('/')[-2]
        base_name = os.path.basename(test_image_path)
    
        # get label info
        image_args = base_name.replace('.jpg','').split('_')
    
        # get camera info
        intrinsic = np.asarray(camera_dict[image_dir]['intrinsics'])
        extrinsic = np.asarray(camera_dict[image_dir]['extrinsics'])
        offset = extrinsic[:, -1]
        rotation = extrinsic[:, :3]

        f = [intrinsic[0][0], intrinsic[1][1]]  # focal length
        c = [intrinsic[0][2], intrinsic[1][2]]  # principal point
        image_scale = 1920
        # predict metrabs
        image = tf.image.decode_jpeg(tf.io.read_file(test_image_path))
        tf_intrinsic = intrinsic.copy()
        tf_intrinsic[:2, :] = tf_intrinsic[:2, :] * image_scale
        intrinsics = tf.constant(tf_intrinsic, dtype=tf.float32)
    
        # get estimated 2d pose coordinates and person boxes
        detections, _, poses2d = metrabs_combined_model.predict_single_image(
            image[CROP_Y:, CROP_X:IMG_SCALE-CROP_X, :], 
            intrinsics, 
            detector_threshold=0.005, 
            detector_nms_iou_threshold=0.01)

        if len(poses2d) == 0:
            detections, _, poses2d = metrabs_combined_model.predict_single_image(
                image[CROP_Y:, CROP_X:IMG_SCALE-CROP_X, :], 
                intrinsics, 
                detector_threshold=0.0001, 
                detector_nms_iou_threshold=0.01)

            print('metrabs predict error : {}'.format(test_image_path))
            person_boxes = [[CROP_X, CROP_Y, IMG_SCALE-(2*CROP_X), 1080-CROP_Y]]        

        # detected person
        person_boxes = detections[0:1, :4].numpy() # x, y, w, h
        person_boxes[:, 0] += CROP_X
        person_boxes[:, 1] += CROP_Y
        tf_person_boxes = tf.constant(person_boxes, tf.float32)
    
        # get 3d pose coordinates
        pred_cam_coord = metrabs_single_model.predict_single_image(image, intrinsics, tf_person_boxes).numpy()[0]
        pred_xyz, pred_uv, pred_scale = cam2pixel(pred_cam_coord, f, c, IMG_SCALE)
        pred_XYZ = cam2world(pred_cam_coord, rotation, offset)
 
        metrabs_info = {
         'img_path' : os.path.join(image_dir, base_name),
         'person_box' : person_boxes[0],
         'metrabs_joint_3d' : pred_XYZ,
         'metrabs_joint_2d' : pred_xyz,
         'metrabs_uv' : pred_uv,
         'metrabs_scale' : pred_scale,
         'intrinsic' : intrinsic,
         'extrinsic' : extrinsic,
         'rotation' : rotation,
         'offset' : offset
        }
        metrabs_db.append(metrabs_info)
    torch.save(metrabs_db, os.path.join(DATA_DIR, 'test_info.pt'))
   

if __name__ == '__main__':

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    train_depth_max = 1680.704895171335 #1st submission

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set test result directory
    make_directory(PERFORMANCE_RECORD_DIR)

    # Set system logger
    system_logger = get_logger(name='test',
                               file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'test_log.log'))


    if not os.path.isfile(os.path.join(DATA_DIR, 'test_info.pt')):
        make_test_meta()
    test_db = torch.load(os.path.join(DATA_DIR, 'test_info.pt'))
    test_dataset = CustomDataset(data_dir=DATA_DIR, mode='test', input_shape=INPUT_SHAPE, output_depth=OUTPUT_DEPTH,
                                       db=test_db, model_type=MODEL_TYPE, depth_max=train_depth_max, coord=COORD)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    
    print('Test set samples:',len(test_dataset))

    # Load Model
    joint_num = 24
    model = get_pose_net(RESNET_TYPE, OUTPUT_DEPTH, OUTPUT_SHAPE, is_train=False, joint_num=joint_num).to(device)

    # Set optimizer, scheduler, loss function, metric function
    metric_fn = get_metric_fn
    model_path = sorted(glob.glob('./results/train/**/best_*.pt'))[-1]
    print(model_path)
    states = torch.load(model_path)
    model.load_state_dict(states['model'])

    # Set tester
    tester = Trainer(model, device, metric_fn, train_depth_max, optimizer=None, scheduler=None, logger=system_logger)

    # test
    submission = pd.read_json('sample_submission.json')
    for epoch_index in tqdm(range(EPOCHS)):
        tester.test_epoch(test_dataloader, epoch_index, MODEL_TYPE, INPUT_SHAPE, OUTPUT_DEPTH, submission)

