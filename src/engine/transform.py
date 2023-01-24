# Global imports
import numpy as np
import torch
import torchvision
import albumentations as albu
import cv2

# Package imports
from osr.engine import albu_transform as albu_fork


# xywh -> x1y1x2y2
def _bbox_transform(bboxes):
    if len(bboxes) == 0:
        bboxes = torch.empty((0, 4), dtype=torch.float)
    else:
        bboxes = torch.FloatTensor(bboxes)
        bboxes[:, 2:] += bboxes[:, :2]
    return bboxes


# Standardize data
def _normalize(image, mean, std):
    image = image / 255.0
    mean = np.array(mean)
    std = np.array(std)
    return ((image - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


# Wrapper class for handling interface with albumentations
class AlbuWrapper(object):
    def __init__(self, albu_transform, stat_dict):
        self.albu_transform = albu_transform
        self.stat_dict = stat_dict
        self.img_transform = torchvision.transforms.ToTensor()
        self.to_key_dict = {
            'boxes': 'bboxes',
            'labels': 'category_ids',
        }
        self.from_key_dict = {v:k for k,v in self.to_key_dict.items()}
        self.from_transform_dict = {
            'category_ids': torch.LongTensor,
            'person_id': torch.LongTensor,
            'bboxes': _bbox_transform,
            'iou_thresh': torch.FloatTensor,
            'id': torch.LongTensor,
            'is_known': torch.BoolTensor,
        }

    def __call__(self, data): 
        # Wrap data into format for albumentations
        image, target = data
        
        # Make sure incoming dimensions match
        assert target['boxes'].size(0) == target['labels'].size(0) == len(target['person_id']), 'Incoming augmentation dimension mismatch'

        #
        rekeyed_target = {(self.to_key_dict[k] if k in self.to_key_dict else k):v for k,v in target.items()}

        #
        albu_result = self.albu_transform(image=_normalize(np.array(image), self.stat_dict['mean'], self.stat_dict['std']), **rekeyed_target)
        new_image = self.img_transform(albu_result['image'])
        new_target = {}
        for k,v in albu_result.items():
            if k != 'image':
                new_target[self.from_key_dict[k] if k in self.from_key_dict else k] = self.from_transform_dict[k](v) if k in self.from_transform_dict else v

        #
        return new_image, new_target


# Random Resized Crop augmentation
def get_transform_rrc(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    if train:
        transform_list = [
            albu_fork.WindowResize(min_size=min_size, max_size=max_size),
            albu.PadIfNeeded(min_width=crop_res, min_height=crop_res, border_mode=cv2.BORDER_CONSTANT, value=0.0),
            albu.OneOf([
                albu_fork.RandomFocusedCrop(height=crop_res, width=crop_res, p=rfc_prob),
                albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
            ], p=1),
            albu.HorizontalFlip(p=0.5),
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)


# Random Resized Crop augmentation
def get_transform_rrc2(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    if train:
        # Calculate padding value
        mean_arr = np.array(stat_dict['mean'])
        std_arr = np.array(stat_dict['std'])
        pad_arr = - mean_arr / std_arr
        #
        transform_list = [
            albu_fork.WindowResize(min_size=min_size, max_size=max_size),
            albu.PadIfNeeded(min_width=3*crop_res, min_height=3*crop_res, border_mode=cv2.BORDER_CONSTANT, value=pad_arr),
            albu.OneOf([
                albu_fork.RandomFocusedCrop2(height=crop_res, width=crop_res, p=rfc_prob),
                albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
            ], p=1),
            albu.HorizontalFlip(p=0.5),
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.6,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)


# Window Resize augmentation
def get_transform_wrs(train, stat_dict, min_size=900, max_size=1500):
    transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    if train:
        transform_list = [
            albu_fork.WindowResize(min_size=min_size, max_size=max_size),
            albu.HorizontalFlip(p=0.5),
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)
