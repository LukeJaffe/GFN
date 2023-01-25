# Global imports
import os
import json
import collections
import numpy as np
import torch
import torchvision
import pandas as pd


#
def get_coco(root, dataset_name, image_set, transforms):
    # Image folder depends on dataset
    if dataset_name == 'cuhk':
        img_folder = 'Image/SSM'
    elif dataset_name == 'prw':
        img_folder = 'frames'

    # Dict of paths
    PATHS = {
        'trainval': 'coco/trainval.json',
        'test': 'coco/test.json',
        'train': 'coco/train.json',
        'val': 'coco/val.json',
        'minitrain': 'coco/minitrain.json',
        'minival': 'coco/minival.json',
        'tinytrainval4': 'coco/tinytrainval4.json',
        'tinytrainval8': 'coco/tinytrainval8.json',
        'tinytrainval12': 'coco/tinytrainval12.json',
        'tinytrainval16': 'coco/tinytrainval16.json',
    }

    # Get anno file
    ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    # Load annotation file
    with open(ann_file, 'r') as fp:
        ann_dict = json.load(fp)

    # Load images
    img_set = set()
    for i, img in enumerate(ann_dict['images']):
        img_set.add(img['id'])
    img_list = sorted(list(img_set))

    # Load annos
    pid_set, uid_set = set(), set()
    pid_img_set_dict = collections.defaultdict(set)
    for i, ann in enumerate(ann_dict['annotations']):
        puid = ann['person_id']
        is_known = ann['is_known']
        image_id = ann['image_id']
        pid_img_set_dict[puid].add(image_id)
        if is_known:
            pid_set.add(puid)
        else:
            uid_set.add(puid)
    pid_list, uid_list = sorted(list(pid_set)), sorted(list(uid_set))
    pid_lookup_dict = {
        'pid_lookup': {},
        'num_img': len(img_list),
        'num_pid': len(pid_list),
        'num_uid': len(uid_list),
    }

    # Create lookup
    for idx, pid in enumerate(pid_list, 1):
        pid_lookup_dict['pid_lookup'][pid] = idx

    # Show partition information
    info_dict = {image_set: {
        '# Images': pid_lookup_dict['num_img'],
        '# Known ID': pid_lookup_dict['num_pid'],
        '# Unknown ID': pid_lookup_dict['num_uid'],
    }}
    info_df = pd.DataFrame(info_dict).T
    print(info_df)

    # Load special dataset loading transform
    t = [ConvertCoco(pid_lookup_dict)]

    if transforms is not None:
        t.append(transforms)
    transforms = torchvision.transforms.Compose(t)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    # Hacky: remove images without annotations if partition has 'train' string in it
    if 'train' in image_set:
        len_before = len(dataset)
        dataset = _remove_images_without_annotations(dataset)
        len_after = len(dataset)
        print('==> Removed images in "{}" without annotations: {}/{}'.format(
            image_set, len_before - len_after, len_before))

    return dataset, pid_lookup_dict


class TestSampler(torch.utils.data.Sampler):
    def __init__(self, partition_name, dataset, retrieval_dir, retrieval_name_list):
        # If dataset is subset, get dataset object
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        # Store params
        self.partition_name = partition_name
        self.dataset = dataset
        self.retrieval_dir = retrieval_dir
        self.retrieval_name_list = retrieval_name_list
        image_id_set = set()
        if 'all' in retrieval_name_list:
            # List of all image_id that we need to gather detects and/or GT features for
            self.image_idx_list = range(len(self.dataset))
            # List of all anno id that we need to gather GT features for
            self.query_id_list = [int(x) for x in list(dataset.coco.anns.keys())]
        else:
            query_id_set, image_id_set = set(), set()
            for retrieval_name in retrieval_name_list:
                retrieval_path = os.path.join(retrieval_dir, '{}.json'.format(retrieval_name))
                with open(retrieval_path, 'r') as fp:
                    retrieval_dict = json.load(fp)
                    _image_id_set = set(retrieval_dict['images'])
                    image_id_set = image_id_set.union(_image_id_set) 
                    # NOTE: retrieval_dict['queries'] can be either a dict or a list, but this is correct in either case
                    _query_id_set = set(retrieval_dict['queries'])
                    query_id_set = query_id_set.union(_query_id_set)
                    print(retrieval_name, len(_image_id_set), len(image_id_set))
                    # The retrieval dict can be large, so delete it to free up space
                    del retrieval_dict
            # List of all image_id that we need to gather detects and/or GT features for
            image_id_list = list(image_id_set)
            self.image_idx_list = [dataset.ids.index(_id) for _id in image_id_list]
            self.query_id_list = [int(x) for x in list(query_id_set)]

    def __iter__(self):
        for image_idx in self.image_idx_list:
            yield image_idx

    def __len__(self):
        return len(self.image_idx_list)


#
class ConvertCoco(object):
    def __init__(self, label_lookup_dict):
        self.label_lookup_dict = label_lookup_dict

    def __call__(self, data):
        image, target = data
        w, h = image.size

        image_id = target['image_id']
        image_id = torch.tensor([image_id])

        anno = target['annotations']

        boxes = [obj['bbox'] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Convert data to tensors
        classes = [obj['category_id'] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        ids = [obj['id'] for obj in anno]
        ids = torch.tensor(ids, dtype=torch.int64)
        iou_thresh = [obj['iou_thresh'] for obj in anno]
        iou_thresh = torch.tensor(iou_thresh, dtype=torch.float32)
        is_known = [obj['is_known'] for obj in anno]
        is_known = torch.tensor(is_known, dtype=torch.bool)

        # Build mask of valid bboxes to keep
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        # Convert boxes to x, y, w, h
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        # Apply valid bbox keep mask to all data
        classes = classes[keep]
        ids = ids[keep]
        iou_thresh = iou_thresh[keep]
        is_known = is_known[keep]

        # Create target dict
        target = {}
        target['boxes'] = boxes
        target['image_id'] = image_id

        # for conversion to coco api
        area = torch.tensor([obj['area'] for obj in anno])

        # Keep only valid PIDs
        try:
            person_id_arr = np.array([obj['person_id'] for obj in anno], dtype=object)[keep.numpy()]
        except IndexError:
            raise Exception({'keep': keep})

        # Store info in target dict
        target['area'] = area
        target['person_id'] = person_id_arr.tolist()
        target['image_size'] = torch.FloatTensor([w, h])
        target['id'] = ids.tolist()
        target['iou_thresh'] = iou_thresh
        target['is_known'] = is_known

        # Store OIM label
        if self.label_lookup_dict is not None:
            target['labels'] = torch.LongTensor([self.label_lookup_dict['pid_lookup'][pid] if pid in self.label_lookup_dict['pid_lookup'] else self.label_lookup_dict['num_pid']+1 for pid in target['person_id']])
        else:
            target['labels'] = classes

        # Check lens are valid
        assert len(target['boxes']) == len(target['labels']) == len(target['person_id'])

        # Check everything for correct dimensions
        assert boxes.size(0) == classes.size(0) == person_id_arr.shape[0]

        return (image, target)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            data = (img, target)
            out = self._transforms(data)
            return out
        else:
            raise Exception


def _remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        else:
            return True

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj['category_id'] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset
