"""Dataset 클래스 정의

TODO:

NOTES:

UPDATED:
"""

import os
import copy
import cv2
import torch
import sys
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from modules.pose_utils import world2cam, cam2pixel
from PIL import Image
import torchvision.transforms as transforms
import random
import pdb

class CustomDataset(Dataset):
    def __init__(self, data_dir, mode, input_shape, output_depth, db, model_type, depth_max, coord, train_ratio=0.9):
        self.data_dir = data_dir
        self.mode = mode
        self.train_ratio = train_ratio
        self.joint_num = 24
        self.skeleton = ((0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
                    (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22),
                    (21, 23))  # 인접된 관절 좌표 정의
        self.image_scale = 1920
        self.input_shape = input_shape
        self.depth_max = depth_max
        self.output_depth = output_depth # 모델 출력 Hitmap 사이즈 (input_shape, input_shape, D)
        self.db = db
        self.model_type = model_type
        self.coord = coord
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((input_shape,input_shape)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_shape,input_shape)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        data = copy.deepcopy(self.db[index])

        # get values        
        if self.mode == 'test':
            mode_dir = 'task04_test'
        else:
            mode_dir = 'task04_train'
        image_path = os.path.join(self.data_dir, mode_dir, 'images', data['img_path'])

        # get meta data
        x, y, w, h = np.array(data['person_box'], dtype=np.int32) # [num_joints, 3]
        input_scale = data['metrabs_scale']
        intrinsic = data['intrinsic']
        extrinsic = data['extrinsic']
        rotation = extrinsic[:,:3]
        offset = extrinsic[:, 3:]
        Tz = offset[-1]
        input_2d = data['metrabs_joint_2d']
 
        if self.mode == 'train':
            pos_aug = np.random.uniform(-20,20,[1,3])
            input_2d = input_2d - pos_aug

        input_s = input_scale - Tz
        input_2d[:, 0:1] = (input_2d[:, 0:1] - x)/w # 0~1 normalize
        input_2d[:, 1:2] = (input_2d[:, 1:2] - y)/h # 0~1 normalize
        input_s = (input_s/self.depth_max + 1.0)/2. # 0~1 normalize

        if self.mode is not 'test':
            gt_2d = data['joint_2d']
            gt_scale = data['scale']
            gt_s = gt_scale - Tz

            if 'heatmap' in self.model_type:         
                # heatmap predict scaled image coordinate
                #gt_2d[:, 0:1] = (gt_2d[:, 0:1] - x)/w*self.input_shape - input_2d[:, 0:1]*self.input_shape# 0 ~ input_shape
                #gt_2d[:, 1:2] = (gt_2d[:, 1:2] - y)/h*self.input_shape - input_2d[:, 1:2]*self.input_shape# 0 ~ input_sahpe
                #gt_s = (gt_s/self.depth_max + 1.0)/2.*self.output_depth - input_s*self.output_depth# 0 ~ output_depth            
                gt_2d[:, 0:1] = (gt_2d[:, 0:1] - x)/w*self.input_shape 
                gt_2d[:, 1:2] = (gt_2d[:, 1:2] - y)/h*self.input_shape 
                gt_s = (gt_s/self.depth_max + 1.0)/2.*self.output_depth             
            else:
                # predict uv coordinate
                gt_2d[:, 0:1] = (gt_2d[:, 0:1] - x)/w # 0~1 normalize
                gt_2d[:, 1:2] = (gt_2d[:, 1:2] - y)/h # 0~1 normalize
                gt_s = (gt_s/self.depth_max + 1.0)/2. # 0~1 normalize
           
            gt_norm_coord = np.concatenate((gt_2d[:,:2], gt_s[:, np.newaxis]), axis=-1)

        input_norm_coord = np.concatenate((input_2d[:,:2], input_s[:, np.newaxis]), axis=-1)
        if not 'img' in self.model_type:
            if self.mode != 'test':
                return self.db[index]['joint_2d'], input_norm_coord, gt_norm_coord, data['joint_3d'], intrinsic, Tz, gt_scale, rotation, offset, np.array(data['person_box'], dtype=np.int32)
            else:
                return image_path, input_norm_coord, intrinsic, Tz, rotation, offset, np.array(data['person_box'], dtype=np.int32)

        else:
            # 1. load image
            input_image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) # H, W, 3
            if not isinstance(input_image, np.ndarray):
                raise IOError("Fail to read %s" % image_path)
            if y+h > 1080:
                h = 1080 - y

            if x+w > 1920:
                w = 1920 - x
            input_image = input_image[y:y+h, x:x+w, :]
            img_patch = self.transform(Image.fromarray(input_image))
            """
            img_patch = []
            for joint in input_2d:
                x = int(joint[0] - self.input_shape/2)
                y = int(joint[1] - self.input_shape/2)
                if x < 0:
                    x = 0
                elif x + self.input_shape > 1920:
                    x = 1920 - self.input_shape
                if y < 0:
                    y = 0
                elif y + self.input_shape > 1080:
                    y = 1080 - self.input_shape

                # get image patch from input joint 
                img_patch.append(input_image[:,y:y+self.input_shape, x:x+self.input_shape])
  
            # 2. crop patch from img & generate patch joint ground truth
            img_patch = np.asarray(img_patch) # num_joints, 3, 256, 256
            """
            if self.mode != 'test':
                return img_patch, self.db[index]['joint_2d'], input_norm_coord, gt_norm_coord, data['joint_3d'], intrinsic, Tz, gt_scale, rotation, offset, np.array(data['person_box'], dtype=np.int32)
            else:
                return image_path, img_patch, input_norm_coord, intrinsic, Tz, rotation, offset, np.array(data['person_box'], dtype=np.int32)



