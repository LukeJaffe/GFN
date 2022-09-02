# Global imports
import os
import re
import copy
import json
import collections
import numpy as np
import scipy.io
import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Local imports
from . import coco_utils


def _get_cam_id(img_name):
    match = re.search(r"c\d", img_name).group().replace("c", "")
    return int(match)


def _load_queries(root):
    query_info = os.path.join(root, "query_info.txt")
    with open(query_info, "rb") as f:
        raw = f.readlines()

    _fix_query_bbox_dict = {
        ('c5s2_042630.jpg', (1859, 280, 60, 182)): (1859, 280, 59, 182),
        ('c5s2_082752.jpg', (77, 385, 120, 219)): (77, 385, 119, 219),
        ('c5s2_120599.jpg', (241, 390, 120, 315)): (241, 390, 119, 315),
    }

    query_list = []
    for line in raw:
        linelist = str(line, "utf-8").split(" ")
        pid = int(linelist[0])
        x, y, w, h = (
            float(linelist[1]),
            float(linelist[2]),
            float(linelist[3]),
            float(linelist[4]),
        )
        roi = np.array([x, y, w, h]).astype(np.int32)
        roi = np.clip(roi, 0, None)  # several coordinates are negative
        roi = tuple(roi.tolist())
        img_name = linelist[5][:-2] + ".jpg"

        # fix inconsistency with bbox loading rounding error
        if (img_name, roi) in _fix_query_bbox_dict:
            roi = _fix_query_bbox_dict[(img_name, roi)]

        #
        query_list.append((img_name, pid, roi))
    return query_list


def _load_split_img_names(root, split):
    """
    Load the image names for the specific split.
    """
    assert split in ("train", "test")
    if split == "train":
        imgs = scipy.io.loadmat(os.path.join(root, "frame_train.mat"))["img_index_train"]
    else:
        imgs = scipy.io.loadmat(os.path.join(root, "frame_test.mat"))["img_index_test"]
    return [img[0][0] + ".jpg" for img in imgs]


def get_partition(root):
    partition_dict = collections.defaultdict(dict)
    cam_id_dict = {}
    unk_count = 5555
    for split in ('train', 'test'):
        imgs = _load_split_img_names(root, split)
        for img_name in imgs:
            anno_path = os.path.join(root, "annotations", img_name)
            anno = scipy.io.loadmat(anno_path)
            box_key = "box_new"
            if box_key not in anno.keys():
                box_key = "anno_file"
            if box_key not in anno.keys():
                box_key = "anno_previous"

            ids = anno[box_key][:, 0].astype(np.int32)
            rois = anno[box_key][:, 1:].astype(np.int32)
            rois = np.clip(rois, 0, None)  # several coordinates are negative

            assert len(rois) == len(ids)

            #rois[:, 2:] += rois[:, :2]
            unk_mask = ids == -2
            num_unk = unk_mask.sum().item()
            ids[ids == -2] = np.arange(unk_count, unk_count + num_unk)
            unk_count += num_unk
            is_known = ~unk_mask

            #
            partition_dict[split][img_name] = list(zip(rois.tolist(), ids.tolist(), is_known.tolist()))
            cam_id_dict[img_name] = _get_cam_id(img_name)

    return partition_dict, cam_id_dict


# Parse a retrieval partition from the CUHK dataset
def parse_retrieval(dataset_dir, image_dict, anno_dict, image_lookup, anno_lookup):
    retrieval_dict = {}
    #
    cam_id_lookup = {}
    for image in image_dict['test']:
        cam_id_lookup[image['id']] = image['cam_id']
    #
    person_id_lookup = {}
    image_id_lookup = {}
    for anno in anno_dict['test']:
        person_id_lookup[anno['id']] = anno['person_id']
        image_id_lookup[anno['id']] = anno['image_id']
    #
    query_list = _load_queries(dataset_dir)
    #
    query_id_list = []
    #
    for image_file, person_id, bbox in query_list:
        anno_id = anno_lookup[(image_file, bbox)]
        assert person_id_lookup[anno_id] == person_id
        query_id_list.append(anno_id)
    _retrieval_dict = {}
    # Load all image files in test set
    file_list = _load_split_img_names(dataset_dir, 'test')
    # Get image_id for each test image file
    image_id_set = set([image_lookup[f] for f in file_list])
    image_id_list = list(image_id_set)
    #
    _retrieval_dict['queries'] = query_id_list
    _retrieval_dict['images'] = image_id_list
    retrieval_dict['test'] = _retrieval_dict

    # prep cam_id retrieval
    _cam_same_retrieval_dict = collections.defaultdict(lambda : collections.defaultdict(list))
    _cam_same_retrieval_dict['images'] = image_id_list
    _cam_cross_retrieval_dict = collections.defaultdict(lambda : collections.defaultdict(list))
    _cam_cross_retrieval_dict['images'] = image_id_list
    for query_id in query_id_list:
        query_image_id = image_id_lookup[query_id]
        query_cam_id = cam_id_lookup[query_image_id]
        for image_id in image_id_set:
            image_cam_id = cam_id_lookup[image_id]
            if query_cam_id == image_cam_id:
                _cam_same_retrieval_dict['queries'][query_id].append(image_id)
            else:
                _cam_cross_retrieval_dict['queries'][query_id].append(image_id)
    retrieval_dict['same_cam_id'] = _cam_same_retrieval_dict 
    retrieval_dict['cross_cam_id'] = _cam_cross_retrieval_dict 

    #
    return retrieval_dict


def _build_adj_mat(trainval_dict, ignore_k_common=100):
    # Ignore the most common pids when building the adjacency matrix
    ## Otherwise, every image is adjacent to every other image because of dense web of connections for PRW
    pid_image_counter = collections.Counter()
    for i, ann in enumerate(trainval_dict['annotations']):
        pid = ann['person_id']
        pid_image_counter[pid] += 1
    most_common_pid_list = [mc[0] for mc in pid_image_counter.most_common(ignore_k_common)]

    # Get set of person_ids associated with each image
    img_id_set_dict = {}
    known_pid_set = set()
    for i, ann in enumerate(trainval_dict['annotations']):
        iid = ann['image_id']
        if iid not in img_id_set_dict:
            img_id_set_dict[iid] = set()
        pid = ann['person_id']
        if pid not in most_common_pid_list:
            img_id_set_dict[iid].add(pid)
            
    # Build adjacency matrix indicating which images share at least one person_id in common
    img_id_list = list(img_id_set_dict.keys())
    n = len(img_id_set_dict)
    adj_mat = np.zeros((n, n), dtype=np.uint8)
    for idx_i, (img_id_i, img_pid_set_i) in tqdm.tqdm(enumerate(img_id_set_dict.items()), total=n):
        for idx_j, (img_id_j, img_pid_set_j) in enumerate(img_id_set_dict.items()):
            if img_id_i != img_id_j:
                if len(img_pid_set_i.intersection(img_pid_set_j)) > 0:
                    adj_mat[idx_i, idx_j] = 1 

    # Build graph from adjacency matrix
    return img_id_list, adj_mat


def _split_trainval(adj_mat, img_id_list, val_frac=0.2, extras=True, seed=2):
    # Store partitions in dictionary
    partition_dict = {}
    
    # Build graph from adjacency matrix
    graph = csr_matrix(adj_mat)
    
    # Get connected components
    n_cc, cc_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    
    #
    cc_dict = collections.defaultdict(list)
    for img_id, cc_label in zip(img_id_list, cc_labels):
        cc_dict[cc_label].append(img_id)
        
    #
    n_val_tot = int(np.ceil((len(img_id_list) * val_frac)))
    n_val = 0

    # Random seed will likely affect if actual frac train/val is remotely close to the target
    np.random.seed(seed)
    rand_cc_labels = np.random.permutation(n_cc)
    train_img_id_list, val_img_id_list = [], []
    for cc_label in rand_cc_labels[::-1]:
        cc = cc_dict[cc_label]
        if n_val < n_val_tot:
            n_val += len(cc)
            val_img_id_list.extend(cc)
        else:
            train_img_id_list.extend(cc)

    # Store train and val partitions
    partition_dict['train'] = train_img_id_list
    partition_dict['val'] = val_img_id_list
    
    # Print actual frac train, frac val
    print('actual frac train: {:.2f}'.format(len(train_img_id_list) / (len(train_img_id_list) + len(val_img_id_list))))
    print('actual frac val: {:.2f}'.format(len(val_img_id_list) / (len(train_img_id_list) + len(val_img_id_list))))
    
    # Build some extra partitions
    if extras:
        # Collect some hardcoded elements from specific connected components
        for cc_label in rand_cc_labels:
            if len(cc_dict[cc_label]) == 3517:
                cc_list1 = cc_dict[cc_label]
            elif len(cc_dict[cc_label]) == 518:
                cc_list2 = cc_dict[cc_label]
            elif len(cc_dict[cc_label]) == 53:
                cc_list3 = cc_dict[cc_label]
                
        # minitrain(100), minival(100): for fast training, eval
        partition_dict['minitrain'] = cc_list1[:100] 
        partition_dict['minival'] = cc_list2[:100] 

        # tinytrainval(4, 8, 12, 16): single batch training for overfitting tests
        partition_dict['tinytrainval4'] = cc_list3[0:4] 
        partition_dict['tinytrainval8'] = cc_list3[4:12] 
        partition_dict['tinytrainval12'] = cc_list3[12:24] 
        partition_dict['tinytrainval16'] = cc_list3[24:40] 
        
        for k in partition_dict:
            print(k, len(partition_dict[k]))
    
    #
    return partition_dict


def _save_partitions(trainval_dict, partition_dict, partition_dir):
    # Iterate through all partitions
    for partition_name, image_id_list in partition_dict.items():
        # Figure out which images to keep
        keep_image_list = []
        for image in trainval_dict['images']:
            if image['id'] in image_id_list:
                keep_image_list.append(image)

        # Figure out which annotations to keep
        keep_anno_list = []
        for anno in trainval_dict['annotations']:
            image_id = anno['image_id']
            if image_id in image_id_list:
                keep_anno_list.append(anno)

        # Build new anno dict
        new_anno_dict = copy.deepcopy(trainval_dict)
        new_anno_dict['images'] = keep_image_list
        new_anno_dict['annotations'] = keep_anno_list
        print(len(keep_image_list), len(keep_anno_list))

        # Save new anno dict
        new_anno_path = os.path.join(partition_dir, '{}.json'.format(partition_name))
        with open(new_anno_path, 'w') as fp:
            json.dump(new_anno_dict, fp)


# Partition trainval set into train and val sets with no common pids
def _partition_trainval_coco(trainval_coco_dict, coco_dir):
    #
    image_id_list, adj_mat = _build_adj_mat(trainval_coco_dict)
    #   
    partition_dict = _split_trainval(adj_mat, image_id_list)
    #
    _save_partitions(trainval_coco_dict, partition_dict, coco_dir)


def prw2coco(dataset_dir):
    # Create base COCO dict
    coco_dict = {
        'info': {
            'description': 'PRW Person Search Dataset',
            'url': 'http://zheng-lab.cecs.anu.edu.au/Project/project_prw.html',
            'version': '1.0',
            'year': 2016,
            'contributor': 'PRW',
            'date_created': '2016/unk', 
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [{
            'supercategory': 'person',
            'id': 1,
            'name': 'person'
        }],
    }

    # Create dir to store coco dataset
    coco_dir = os.path.join(dataset_dir, 'coco')
    if not os.path.exists(coco_dir):
        os.makedirs(coco_dir)

    # Create dir to store coco dataset
    retrieval_json_dir = os.path.join(dataset_dir, 'retrieval')
    if not os.path.exists(retrieval_json_dir):
        os.makedirs(retrieval_json_dir)

    # Copy coco dict
    trainval_coco_dict = copy.deepcopy(coco_dict)
    test_coco_dict = copy.deepcopy(coco_dict)

    # Get partition dict
    partition_dict, cam_id_dict = get_partition(dataset_dir) 

    # Get CUHK image and annotation metadata
    image_dir = os.path.join(dataset_dir, 'frames')
    image_dict, anno_dict, image_lookup, anno_lookup = coco_utils.get_coco_dict(
        partition_dict, image_dir, cam_id_dict=cam_id_dict)

    # Parse test retrieval scenarios
    retrieval_dict = parse_retrieval(dataset_dir, image_dict, anno_dict, image_lookup, anno_lookup)
    for _retrieval_name, _retrieval_dict in retrieval_dict.items():
        retrieval_json_file = '{}.json'.format(_retrieval_name)
        retrieval_json_path = os.path.join(retrieval_json_dir, retrieval_json_file)
        with open(retrieval_json_path, 'w') as fp:
            json.dump(_retrieval_dict, fp)

    # Store CUHK metadata in COCO dicts
    trainval_coco_dict['images'] = image_dict['train']
    trainval_coco_dict['annotations'] = anno_dict['train']
    test_coco_dict['images'] = image_dict['test']
    test_coco_dict['annotations'] = anno_dict['test']

    # Save COCO dicts as JSONs in coco dir
    trainval_anno_path = os.path.join(coco_dir, 'trainval.json')
    test_anno_path = os.path.join(coco_dir, 'test.json')
    with open(trainval_anno_path, 'w') as fp:
        json.dump(trainval_coco_dict, fp)
    with open(test_anno_path, 'w') as fp:
        json.dump(test_coco_dict, fp)

    # Partition trainval into train and val
    _partition_trainval_coco(trainval_coco_dict, coco_dir)

    # Build val retrieval set
    val_anno_path = os.path.join(coco_dir, 'val.json')
    with open(val_anno_path, 'r') as fp:
        val_coco_dict = json.load(fp)
    coco_utils.build_coco_retrieval(retrieval_json_dir, 'val', val_coco_dict)


# Main function
def main():
    # Function imports
    import argparse

    # Parse user args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/datasets/prw')
    args = parser.parse_args() 

    # Build the COCO dataset for CUHK
    prw2coco(args.dataset_dir)
