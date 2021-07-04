"""Trainer 클래스 정의

TODO:

NOTES:

REFERENCE:

UPDATED:
"""


import torch
import tqdm
import pdb
import numpy as np
import os
from modules.utils import load_yaml, save_json
class Trainer():
    """ Trainer
        epoch에 대한 학습 및 검증 절차 정의
    """

    def __init__(self, model, device, metric_fn, depth_max, optimizer=None, scheduler=None, logger=None):
        """ 초기화
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self.metric_fn = metric_fn
        self.depth_max = depth_max

    def get_world_coord(self, norm_coord, person_boxes, image_scale, depth_max, Tz, rotation, intrinsic, offset):
        norm_coord[:, :, 0] = (norm_coord[:,:,0] * person_boxes[:, 2:3] + person_boxes[:, 0:1])/image_scale # u = (* width + x_pos)/image_scale
        norm_coord[:, :, 1] = (norm_coord[:,:,1] * person_boxes[:, 3:4] + person_boxes[:, 1:2])/image_scale # v = (* height + y_pos)image_scale
        norm_coord[:, :, 2] = (norm_coord[:,:,2] * 2.0 - 1.0) * depth_max + Tz # s = (*2.0 -1.0)*depth_max + Tz
        norm_coord[:, :, 0:2] = norm_coord[:, :, 0:2] * norm_coord[:, :, 2:3]
        uvs = norm_coord.numpy()
        inv_rotation = np.linalg.inv(rotation.numpy())
        inv_intrinsic = np.linalg.inv(intrinsic.numpy())
        world_coord = ((inv_rotation @ inv_intrinsic) @ uvs.transpose(0,2,1)).transpose(0,2,1) - (inv_rotation @ offset.numpy()).transpose(0,2,1)
        return world_coord
        
    def train_epoch(self, dataloader, epoch_index, model_type, input_shape, output_depth):
        """ 한 epoch에서 수행되는 학습 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.train()
        self.train_total_loss = 0
        self.train_score = 0
        
        pbar = tqdm.tqdm(dataloader)

        for batch_index, inputs in enumerate(pbar):
            if 'img' in model_type:
                img_patch, gt_2d, input_norm_coord, gt_norm_coord, gt_world_coord, intrinsic, Tz, gt_scale, rotation, offset, person_boxes = inputs
                img_patch = img_patch.float().to(self.device)
                input_norm_coord = input_norm_coord.float().to(self.device)
                gt_norm_coord = gt_norm_coord.float().to(self.device)
                pred_norm_coord = self.model(input_norm_coord, img_patch)
            else:
                gt_2d, input_norm_coord, gt_norm_coord, gt_world_coord, intrinsic, Tz, gt_scale, rotation, offset, person_boxes = inputs
                input_norm_coord = input_norm_coord.float().to(self.device)
                gt_norm_coord = gt_norm_coord.float().to(self.device)
                pred_norm_coord = self.model(input_norm_coord, input_norm_coord)

            ## abs loss
            loss_coord = torch.abs(pred_norm_coord - gt_norm_coord)
            loss = loss_coord.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if 'heatmap' in model_type:
                pred_norm_coord[:, :, 0] /= input_shape
                pred_norm_coord[:, :, 1] /= input_shape
                pred_norm_coord[:, :, 2] /= output_depth

            pred_world_coord = self.get_world_coord(pred_norm_coord.detach().cpu(), person_boxes, 1920, self.depth_max, Tz, rotation, intrinsic, offset)
            self.train_total_loss += loss.detach().item()
            world_coord_loss = np.mean(np.mean(np.mean(np.power(np.abs(pred_world_coord - gt_world_coord.numpy()), 2), axis=2), axis=1))
            self.train_score += world_coord_loss
            pbar.set_description('loss {:.4f} world_coord_loss {:4f}'.format(self.train_total_loss/(batch_index+1), world_coord_loss))
            if 'img' in model_type:
                del img_patch, gt_2d, input_norm_coord, gt_norm_coord, gt_world_coord, intrinsic, Tz, gt_scale, rotation, offset, person_boxes
            else:
                del gt_2d, input_norm_coord, gt_norm_coord, gt_world_coord, intrinsic, Tz, gt_scale, rotation, offset, person_boxes
            
        #y_pred = np.concatenate(np.asarray(pred_lst))
        #y_answer = np.concatenate(np.asarray(target_lst))
        self.train_mean_loss = self.train_total_loss / len(dataloader)
        self.train_score = self.train_score / len(dataloader)
        #self.train_score = self.metric_fn(y_pred=torch.Tensor(y_pred), y_answer=torch.Tensor(y_answer))
        learning_rate = self.scheduler.get_last_lr()[0]
        msg = f'Epoch {epoch_index}, Train loss: {self.train_mean_loss}, Score: {self.train_score} LR: {learning_rate}'
        print(msg)
        self.logger.info(msg) if self.logger else print(msg)

    def validate_epoch(self, dataloader, epoch_index, model_type, input_shape, output_depth):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()
        self.val_total_loss = 0
        self.validation_score =0
        pbar = tqdm.tqdm(dataloader)
        with torch.no_grad():
            for batch_index, inputs in enumerate(pbar):
                if 'img' in model_type:
                    img_patch, gt_2d, input_norm_coord, gt_norm_coord, gt_world_coord, intrinsic, Tz, gt_scale, rotation, offset, person_boxes = inputs
                    img_patch = img_patch.float().to(self.device)
                    input_norm_coord = input_norm_coord.float().to(self.device)
                    gt_norm_coord = gt_norm_coord.float().to(self.device)
                    pred_norm_coord = self.model(input_norm_coord, img_patch)
                else:
                    gt_2d, input_norm_coord, gt_norm_coord, gt_world_coord, intrinsic, Tz, gt_scale, rotation, offset, person_boxes = inputs
                    input_norm_coord = input_norm_coord.float().to(self.device)
                    gt_norm_coord = gt_norm_coord.float().to(self.device)
                    pred_norm_coord = self.model(input_norm_coord, input_norm_coord)
            
                ## abs loss
                loss_coord = torch.abs(pred_norm_coord - gt_norm_coord)
                loss = loss_coord.mean()
                if 'heatmap' in model_type:
                    pred_norm_coord[:, :, 0] /= input_shape
                    pred_norm_coord[:, :, 1] /= input_shape
                    pred_norm_coord[:, :, 2] /= output_depth

                pred_world_coord = self.get_world_coord(pred_norm_coord.detach().cpu(), person_boxes, 1920, self.depth_max, Tz, rotation, intrinsic, offset)
                self.val_total_loss += loss
                world_coord_loss = np.mean(np.mean(np.mean(np.power(np.abs(pred_world_coord - gt_world_coord.numpy()), 2), axis=2), axis=1))
                self.validation_score += world_coord_loss
                pbar.set_description('loss {:.4f} world_coord_loss {:4f}'.format(self.val_total_loss/(batch_index+1), world_coord_loss))
                if 'img' in model_type:
                    del img_patch, gt_2d, input_norm_coord, gt_norm_coord, gt_world_coord, intrinsic, Tz, gt_scale, rotation, offset, person_boxes
                else:
                    del gt_2d, input_norm_coord, gt_norm_coord, gt_world_coord, intrinsic, Tz, gt_scale, rotation, offset, person_boxes

            self.val_mean_loss = self.val_total_loss / len(dataloader)
            self.validation_score = self.validation_score / len(dataloader)
            #self.validation_score = self.metric_fn(y_pred=torch.Tensor(y_pred), y_answer=torch.Tensor(y_answer))
            msg = f'Epoch {epoch_index}, Validation loss: {self.val_mean_loss}, Score: {self.validation_score}'
            print(msg)
            self.logger.info(msg) if self.logger else print(msg)

    def test_epoch(self, dataloader, epoch_index, model_type, input_shape, output_depth, submission):
        """ 한 epoch에서 수행되는 테스트 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()
        target_lst = []
        pred_lst = []
        pbar = tqdm.tqdm(dataloader)
        with torch.no_grad():
            for batch_index, inputs in enumerate(pbar):
                if 'img' in model_type:
                    image_path, img_patch, input_norm_coord, intrinsic, Tz, rotation, offset, person_boxes = inputs
                    img_patch = img_patch.float().to(self.device)
                    input_norm_coord = input_norm_coord.float().to(self.device)
                    pred_norm_coord = self.model(input_norm_coord, img_patch)
                else:
                    image_path, input_norm_coord, intrinsic, Tz, rotation, offset, person_boxes = inputs
                    input_norm_coord = input_norm_coord.float().to(self.device)
                    pred_norm_coord = self.model(input_norm_coord, input_norm_coord)
            
                if 'heatmap' in model_type:
                    pred_norm_coord[:, :, 0] /= input_shape
                    pred_norm_coord[:, :, 1] /= input_shape
                    pred_norm_coord[:, :, 2] /= output_depth
                    #pred_norm_coord += input_norm_coord

                pred_world_coord = self.get_world_coord(pred_norm_coord.detach().cpu(), person_boxes, 1920, self.depth_max, Tz, rotation, intrinsic, offset)
                img_path = image_path[0]
                idx = submission[submission['file_name'] == os.path.basename(img_path)].index[0]
                submission.iloc[idx]['3d_pos'] = pred_world_coord[0].tolist()
        save_json('pred_submission.json', submission) 
        print('save pred_submission.json ')
