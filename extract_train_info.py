import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm
import json
import pdb
from collections import defaultdict
import numpy as np
import cv2
import random
import torch
from PIL import Image

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

def get_mpjpe(y_pred, y_answer):
    mpjpe = np.mean(np.power(y_pred - y_answer, 2), axis=0)
    return mpjpe[0], mpjpe[1], mpjpe[2]

def save_2d_pose(person_boxes, image, poses2d, edges, save_path):
    fig = plt.figure()
    image_ax = fig.add_subplot(1, 1, 1)
    #image_ax.set_title('Input')
    image_ax.imshow(image)
    for idx, pose2d in enumerate(poses2d):
        # Create a Rectangle patch
        x = person_boxes[idx][0]
        y = person_boxes[idx][1]
        w = person_boxes[idx][2]
        h = person_boxes[idx][3]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        image_ax.add_patch(rect)
        rect = patches.Rectangle((440, 200), 1040, 880, linewidth=1, edgecolor='g', facecolor='none')
        image_ax.add_patch(rect)
        for i_start, i_end in edges:
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
        image_ax.scatter(pose2d[:, 0], pose2d[:, 1], s=2)

    #fig.tight_layout()
    plt.savefig(fname=save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

############################################### GET META DATA ################################################
BASE_DIR = '/DATA/Final_DATA'
if os.path.isfile(os.path.join(BASE_DIR, 'train_meta.pt')):
    exit()

import tensorflow as tf
TRAIN_DATA_PATH = '/DATA/Final_DATA/task04_train'
train_image_list = glob.glob(TRAIN_DATA_PATH+'/images/*/*.jpg')
train_camera_list = glob.glob(TRAIN_DATA_PATH+'/camera/*.json')
train_label_list = glob.glob(TRAIN_DATA_PATH+'/labels/*/*.json')

image_dict = defaultdict()
random.shuffle(train_image_list)
for train_image_path in train_image_list:
    train_args = os.path.basename(train_image_path).replace('.jpg','').split('_')
    pose_type = train_args[0]
    person_type = train_args[1]
    camera_type = train_args[2]
    pose_sequence = '{0:04d}'.format(int(train_args[3]))
    image_key = '{}_{}_{}_{}'.format(pose_type, person_type, camera_type, pose_sequence)
    image_dict[image_key] = train_image_path
    
camera_dict = defaultdict()
for train_camera_path in train_camera_list:
    camera_info = open(train_camera_path, 'r')
    camera_json = json.load(camera_info)
    camera_dict[os.path.basename(train_camera_path).split('.json')[0]] = camera_json

label_dict = defaultdict()
for train_label_path in train_label_list:
    label_info = open(train_label_path, 'r')
    label_json = json.load(label_info)
    # ex) 3D_30_F160A_10.json => 30_F160A_10
    label_dict[os.path.basename(train_label_path).split('.json')[0].replace('3D_','')] = label_json

skeleton = ((0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
                     #(9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22),
                     (12, 13), (12, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22),
                     (21, 23))
joint_edges = np.asarray(skeleton)
IMG_SCALE = 1920
CROP_X = 440
CROP_Y = 200
print('total train image : {}'.format(len(train_image_list)))
############################################################################################################


############################################### GET METRABS ################################################
metrabs_model_path = './pre_models'
model_combined_type = 'metrabs_multiperson_smpl_combined'
model_single_type = 'metrabs_multiperson_smpl'
metrabs_combined_model = tf.saved_model.load(os.path.join(metrabs_model_path, model_combined_type))
metrabs_single_model = tf.saved_model.load(os.path.join(metrabs_model_path, model_single_type))
############################################################################################################

pbar = tqdm.tqdm(image_dict.items())
train_db = []

for image_key, train_image_path in pbar:   
   
    image_dir = train_image_path.split('/')[-2]
    base_name = os.path.basename(train_image_path)
    
    # get label info
    image_args = base_name.replace('.jpg','').split('_')
    
    label_key = '{}_{}_{}'.format(image_args[0], image_args[1], image_args[3])
    label_json = label_dict[label_key]
    joint_name_3d = label_json['info']['3d_pos']
    XYZ = np.asarray(label_json['annotations']['3d_pos']).squeeze(axis=-1)
    # get camera info
    intrinsic = np.asarray(camera_dict[image_dir]['intrinsics'])
    # The camera's extrinsic matrix describes the camera's location in the world, and what direction it's pointing.
    extrinsic = np.asarray(camera_dict[image_dir]['extrinsics'])


    # my method : get world coordinate with intrinsic and extrinsic
    offset = extrinsic[:, -1]
    rotation = extrinsic[:, :3]
    f = [intrinsic[0][0], intrinsic[1][1]]  # focal length
    c = [intrinsic[0][2], intrinsic[1][2]]  # principal point
    cam_coord = world2cam(XYZ, rotation, offset)
    xyz, uv, scale = cam2pixel(cam_coord, f, c, IMG_SCALE)

    # predict metrabs
    try:
        image = tf.image.decode_jpeg(tf.io.read_file(train_image_path))
    except:
        print('tf_image read erro : {}'.format(train_image_path))
        continue
         

    tf_intrinsic = intrinsic.copy()
    tf_intrinsic[:2, :] = tf_intrinsic[:2, :] * IMG_SCALE
    intrinsics = tf.constant(tf_intrinsic, dtype=tf.float32)

    # get estimated 2d pose coordinates and person boxes
    detections, _, poses2d = metrabs_combined_model.predict_single_image(
        image[CROP_Y:, CROP_X:IMG_SCALE-CROP_X, :], 
        intrinsics, 
        detector_threshold=0.005, 
        detector_nms_iou_threshold=0.01)

    if len(poses2d) == 0:
        print('metrabs predict error : {}'.format(train_image_path))
        continue

    # detected person
    person_boxes = detections[0:1, :4].numpy() # x, y, w, h
    person_boxes[:, 0] += CROP_X
    person_boxes[:, 1] += CROP_Y
    tf_person_boxes = tf.constant(person_boxes, tf.float32)
    
    # get 3d pose coordinates
    pred_cam_coord = metrabs_single_model.predict_single_image(image, intrinsics, tf_person_boxes).numpy()[0]
    pred_xyz, pred_uv, pred_scale = cam2pixel(pred_cam_coord, f, c, IMG_SCALE)
    pred_XYZ = cam2world(pred_cam_coord, rotation, offset)
    mpjpe_X, mpjpe_Y, mpjpe_Z = get_mpjpe(pred_XYZ, XYZ)
    train_meta = {
     'img_path' : os.path.join(image_dir, base_name),
     'person_box' : person_boxes[0],
     'joint_3d' : XYZ, 
     'metrabs_joint_3d' : pred_XYZ,
     'joint_2d' : xyz, 
     'metrabs_joint_2d' : pred_xyz,
     'scale' : scale,
     'metrabs_scale' : pred_scale,
     'mpjpe_x' : mpjpe_X,
     'mpjpe_y' : mpjpe_Y,
     'mpjpe_z' : mpjpe_Z,
     'intrinsic' : intrinsic,
     'extrinsic' : extrinsic,
     'rotation' : rotation,
     'offset' : offset,
     'focal_length' : f,
     'principal_point' : c,
     'pose_type' : image_args[0],
     'sex' : image_args[1][0],
     'height' : image_args[1][1:4],
     'obesity' : image_args[1][-1],
     'camera_num' : image_args[2]
    }
    train_db.append(train_meta)
print('extract metrabs info done')
torch.save(train_db, os.path.join(BASE_DIR, 'train_meta.pt'))
