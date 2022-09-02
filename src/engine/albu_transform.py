# Global imports
import cv2
import math
import random
import numpy as np
import torch
from typing import Union, Sequence, Optional, Tuple, List, Dict

# albumentations imports
from albumentations.augmentations.crops import functional as F
# albumentations imports dependent on version
try:
    from albumentations.augmentations.bbox_utils import union_of_bboxes
    from albumentations import convert_bbox_from_albumentations
except:
    from albumentations.core.bbox_utils import union_of_bboxes
    from albumentations.core.bbox_utils import convert_bbox_from_albumentations
from albumentations.augmentations.geometric import functional as FGeometric
from albumentations.core.transforms_interface import DualTransform


__all__ = [
    "WindowResize",
    "RandomFocusedCrop",
    "RandomFocusedCrop2",
]

class WindowResize(DualTransform):
    """Rescale an image so that it fits within a fixed size window, keeping the aspect ratio of the initial image.
    Args:
        min_size (int, list of int): minimimum side length of resize window
        max_size (int, list of int): maximum side length of resize window
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        min_size: Union[int, Sequence[int]] = 900,
        max_size: Union[int, Sequence[int]] = 1500,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super(WindowResize, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.min_size = min_size
        self.max_size = max_size

    def apply(
        self, img: np.ndarray, min_size: int = 900, max_size: int = 1500, interpolation: int = cv2.INTER_LINEAR, **params
    ) -> np.ndarray:
        height = params["rows"]
        width = params["cols"]
        image_min_size = min(width, height)
        image_max_size = max(width, height)
        scale_factor = min_size / image_min_size
        if image_max_size * scale_factor > max_size:
            return FGeometric.longest_max_size(img, max_size=max_size, interpolation=interpolation)
        else:
            return FGeometric.smallest_max_size(img, max_size=min_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox: Sequence[float], **params) -> Sequence[float]:
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint: Sequence[float], min_size: int = 900, max_size: int = 1500, **params) -> Sequence[float]:
        raise NotImplementedError
        height = params["rows"]
        width = params["cols"]

        scale = max_size / max([height, width])
        return FGeometric.keypoint_scale(keypoint, scale, scale)

    def get_params(self) -> Dict[str, int]:
        return {
            "min_size": self.min_size if isinstance(self.min_size, int) else random.choice(self.min_size),
            "max_size": self.max_size if isinstance(self.max_size, int) else random.choice(self.max_size),
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("min_size", "max_size", "interpolation")

class _BaseRandomFocusedCrop(DualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(_BaseRandomFocusedCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, xmin=0, ymin=0, xmax=0, ymax=0, scale=1.0, interpolation=cv2.INTER_LINEAR, **params):
        if scale != 1.0:
            img = FGeometric.scale(img, scale, interpolation)
        crop = F.crop(img, xmin, ymin, xmax, ymax)
        assert crop.shape[:2] == (self.height, self.width)
        return crop

    def apply_to_bbox(self, bbox, xmin=0, ymin=0, xmax=0, ymax=0, rows=0, cols=0, scale=1.0, **params):
        cropped_bbox = F.bbox_crop(bbox, xmin, ymin, xmax, ymax, rows*scale, cols*scale)
        return cropped_bbox

class RandomFocusedCrop(_BaseRandomFocusedCrop):
    """Crop a random part of the input and rescale it to some size.

    Args:
        min_max_height ((int, int)): crop size limits.
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        w2h_ratio (float): aspect ratio of crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0
    ):
        super(RandomFocusedCrop, self).__init__(
            height=height, width=width, interpolation=interpolation, always_apply=always_apply, p=p
        )
        self.height = height
        self.width = width

    def get_params_dependent_on_targets(self, params):
        image = params['image']
        h, w, _ = image.shape
        boxes = params['bboxes']
        pids = [box[-1] for box in boxes]
        #pid_mask = [p.startswith('p') for p in pids]
        pid_mask = params['is_known']
        #pid_idx = torch.where(torch.tensor(pid_mask))[0].tolist()
        pid_idx = torch.where(pid_mask)[0].tolist()
        if len(pid_idx) > 0:
            rand_idx = random.choice(pid_idx)
            rand_box = [c.item() for c in boxes[rand_idx][:4]]
            denorm_box = convert_bbox_from_albumentations(rand_box, 'coco', rows=h, cols=w)
            
            # Determine valid limits for the crop
            bx, by, bw, bh = denorm_box
            lxlb = min(math.floor(bx), math.floor(max(0, bx+bw - self.width)))
            lxub = max(lxlb, math.floor(min(w - self.width, bx)))
            lylb = min(math.floor(by), math.floor(max(0, by+bh - self.height)))
            lyub = max(lylb, math.floor(min(h - self.height, by)))
        else:
            lxlb = 0
            lylb = 0
            lxub = w - self.width
            lyub = h - self.height
        
        # Randomly select x and y from limits
        rx = random.randint(lxlb, lxub)
        ry = random.randint(lylb, lyub)

        # Set final crop coordinates
        x1 = int(rx)
        y1 = int(ry)
        x2 = x1 + self.width
        y2 = y1 + self.height
        
        # Put crop coordinates in dict
        return {
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2,
        }

    @property
    def targets_as_params(self):
        return ['image', 'bboxes', 'is_known']
    
    def get_transform_init_args_names(self):
        return "height", "width"

class _BaseRandomFocusedCrop2(DualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(_BaseRandomFocusedCrop2, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, xmin=0, ymin=0, xmax=0, ymax=0, scale=1.0, interpolation=cv2.INTER_LINEAR, **params):
        if scale != 1.0:
            img = FGeometric.scale(img, scale, interpolation)
        crop = F.crop(img, xmin, ymin, xmax, ymax)
        crop = FGeometric.resize(crop, self.height, self.width, interpolation)
        assert crop.shape[:2] == (self.height, self.width)
        return crop

    def apply_to_bbox(self, bbox, xmin=0, ymin=0, xmax=0, ymax=0, rows=0, cols=0, scale=1.0, **params):
        cropped_bbox = F.bbox_crop(bbox, xmin, ymin, xmax, ymax, rows*scale, cols*scale)
        return cropped_bbox

class RandomFocusedCrop2(_BaseRandomFocusedCrop2):
    """Crop a random part of the input and rescale it to some size.

    Args:
        min_max_height ((int, int)): crop size limits.
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        w2h_ratio (float): aspect ratio of crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0
    ):
        super(RandomFocusedCrop2, self).__init__(
            height=height, width=width, interpolation=interpolation, always_apply=always_apply, p=p
        )
        self.height = height
        self.width = width

    def get_params_dependent_on_targets(self, params):
        image = params['image']
        h, w, _ = image.shape
        boxes = params['bboxes']
        pids = [box[-1] for box in boxes]
        #pid_mask = [p.startswith('p') for p in pids]
        pid_mask = params['is_known']
        #pid_idx = torch.where(torch.tensor(pid_mask))[0].tolist()
        pid_idx = torch.where(pid_mask)[0].tolist()
        if len(pid_idx) > 0:
            rand_idx = random.choice(pid_idx)
            rand_box = [c.item() for c in boxes[rand_idx][:4]]
            denorm_box = convert_bbox_from_albumentations(rand_box, 'coco', rows=h, cols=w)
            
            # Determine valid limits for the crop
            bx, by, bw, bh = denorm_box
            if bw > self.width:
                crop_width = bw
            else:
                crop_width = self.width
            if bh > self.height:
                crop_height = bh
            else:
                crop_height = self.height
            max_dim = int(math.floor(max(crop_width, crop_height)))
            crop_width = crop_height = max_dim
            
            lxlb = min(math.floor(bx), math.floor(max(0, bx+bw - crop_width)))
            lxub = max(lxlb, math.floor(min(w - crop_width, bx)))
            lylb = min(math.floor(by), math.floor(max(0, by+bh - crop_height)))
            lyub = max(lylb, math.floor(min(h - crop_height, by)))
        else:
            lxlb = 0
            lylb = 0
            lxub = w - self.width
            lyub = h - self.height
            crop_width = self.width
            crop_height = self.height
        
        # Randomly select x and y from limits
        rx = random.randint(lxlb, lxub)
        ry = random.randint(lylb, lyub)

        # Set final crop coordinates
        x1 = int(rx)
        y1 = int(ry)
        x2 = x1 + crop_width
        y2 = y1 + crop_height
        
        # Put crop coordinates in dict
        return {
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2,
        }

    @property
    def targets_as_params(self):
        return ['image', 'bboxes', 'is_known']
    
    def get_transform_init_args_names(self):
        return "height", "width"
