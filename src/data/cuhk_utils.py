# Global imports
import os
import copy
import json
import collections
import scipy.io
import numpy as np
import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Local imports
from . import coco_utils


def _load_annotations(root, split):
    # load all images and build a dict from image to boxes
    all_imgs = scipy.io.loadmat(os.path.join(root, "annotation", "Images.mat"))
    all_imgs = all_imgs["Img"].squeeze()
    name_to_boxes = {}
    name_to_pids = {}
    c = 5555 # greater than number of labeled pids
    for img_name, _, boxes in all_imgs:
        img_name = str(img_name[0])
        boxes = np.asarray([b[0] for b in boxes[0]])
        boxes = boxes.reshape(boxes.shape[0], 4)  # (x1, y1, w, h)
        valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
        assert valid_index.size > 0, "Warning: {} has no valid boxes.".format(img_name)
        boxes = boxes[valid_index]
        name_to_boxes[img_name] = boxes.astype(np.int32)
        name_to_pids[img_name] = [(x, False) for x in ((np.arange(boxes.shape[0]))+c).tolist()]
        c += boxes.shape[0]

    def set_box_pid(boxes, box, pids, pid):
        for i in range(boxes.shape[0]):
            if np.all(boxes[i] == box):
                pids[i] = (pid, True)
                return

    # assign a unique pid from 1 to N for each identity
    if split == "train":
        train = scipy.io.loadmat(os.path.join(root, "annotation/test/train_test/Train.mat"))
        train = train["Train"].squeeze()
        for index, item in enumerate(train):
            scenes = item[0, 0][2].squeeze()
            for img_name, box, _ in scenes:
                img_name = str(img_name[0])
                box = box.squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[img_name], box, name_to_pids[img_name], index + 1)
    else:
        protoc = scipy.io.loadmat(os.path.join(root, "annotation/test/train_test/TestG50.mat"))
        protoc = protoc["TestG50"].squeeze()
        for index, item in enumerate(protoc):
            # query
            im_name = str(item["Query"][0, 0][0][0])
            box = item["Query"][0, 0][1].squeeze().astype(np.int32)
            set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)
            # gallery
            gallery = item["Gallery"].squeeze()
            for im_name, box, _ in gallery:
                im_name = str(im_name[0])
                if box.size == 0:
                    break
                box = box.squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)
    return name_to_pids, name_to_boxes


### Test set files
def get_partition(dataset_dir):
    test_partition_path = os.path.join(dataset_dir, 'annotation/pool.mat')
    test_partition_arr = scipy.io.loadmat(test_partition_path)['pool'].ravel()
    test_partition_list = [x[0] for x in test_partition_arr]
    test_partition_set = set(test_partition_list)

    anno_path = os.path.join(dataset_dir, 'annotation/Images.mat')
    anno_arr = scipy.io.loadmat(anno_path)['Img'].ravel()
    full_file_list = []
    num_small = 0
    for anno in anno_arr:
        img_file = anno[0][0]
        full_file_list.append(img_file)

    # Split data into partitions
    full_file_set = set(full_file_list)
    train_partition_set = full_file_set - test_partition_set
    partition_file_dict = {
        'train': train_partition_set,
        'test': test_partition_set,
    }

    #
    partition_dict = collections.defaultdict(dict)
    partition_num_boxes = 0
    for mode in partition_file_dict:
        file_to_pids, file_to_boxes = _load_annotations(dataset_dir, mode)
        for img_file in partition_file_dict[mode]:
            pids = file_to_pids[img_file]
            boxes = [tuple(b) for b in file_to_boxes[img_file].tolist()]
            partition_dict[mode][img_file] = list(zip(boxes, *zip(*pids)))

    return partition_dict


def _build_adj_mat(trainval_dict):
    # Get set of person_ids associated with each image
    image_id_set_dict = collections.defaultdict(set)
    for i, ann in enumerate(trainval_dict['annotations']):
        iid = ann['image_id']
        pid = ann['person_id']
        image_id_set_dict[iid].add(pid)
    
    # Build adjacency matrix indicating which images share at least one person_id in common
    image_id_list = list(image_id_set_dict.keys())
    n = len(image_id_set_dict)
    adj_mat = np.zeros((n, n), dtype=np.uint8)
    for idx_i, (image_id_i, image_pid_set_i) in tqdm.tqdm(enumerate(image_id_set_dict.items()), total=n):
        for idx_j, (image_id_j, image_pid_set_j) in enumerate(image_id_set_dict.items()):
            if image_id_i != image_id_j:
                if len(image_pid_set_i.intersection(image_pid_set_j)) > 0:
                    adj_mat[idx_i, idx_j] = 1 

    # Build graph from adjacency matrix
    return image_id_list, adj_mat


def _split_trainval(adj_mat, image_id_list, val_frac=0.2, extras=True, seed=0):
    # Store partitions in dictionary
    partition_dict = {}
    
    # Build graph from adjacency matrix
    graph = csr_matrix(adj_mat)
    # Get connected components
    n_cc, cc_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    
    #
    np.random.seed(seed)
    cc_dict = collections.defaultdict(list)
    for image_id, cc_label in zip(image_id_list, cc_labels):
        cc_dict[cc_label].append(image_id)

    #
    n_val_tot = int(np.ceil((len(image_id_list) * val_frac)))
    n_val = 0

    #
    rand_cc_labels = np.random.permutation(n_cc)
    train_image_id_list, val_image_id_list = [], []
    for cc_label in rand_cc_labels:
        cc = cc_dict[cc_label]
        if (n_val < n_val_tot) and (len(cc) >= 2):
            n_val += len(cc)
            val_image_id_list.extend(cc)
        else:
            train_image_id_list.extend(cc)

    # Store train and val partitions
    partition_dict['train'] = train_image_id_list
    partition_dict['val'] = val_image_id_list
            
    # Build some extra partitions
    if extras:
        # minitrain(100), minival(100): for fast training, eval
        cc_list4 = []
        for cc_label in rand_cc_labels:
            if len(cc_dict[cc_label]) == 4:
                cc_list4.append(cc_dict[cc_label])
        partition_dict['minitrain'] = sum(cc_list4[:25], [])
        partition_dict['minival'] = sum(cc_list4[25:50], [])
        
        # tinytrainval(4, 8, 12, 16): single batch training for overfitting tests
        partition_dict['tinytrainval4'] = sum(cc_list4[50:51], [])
        partition_dict['tinytrainval8'] = sum(cc_list4[51:53], [])
        partition_dict['tinytrainval12'] = sum(cc_list4[53:56], [])
        partition_dict['tinytrainval16'] = sum(cc_list4[56:60], []) 
        
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


# Parse a retrieval partition from the CUHK dataset
def _parse_retrieval(image_lookup, anno_lookup, retrieval_name, retrieval_path):
    retrieval_mat = scipy.io.loadmat(retrieval_path)
    # Determine dict key for the retrieval data in the matrix
    _retrieval_key = os.path.basename(retrieval_path).split('.')[0]
    if _retrieval_key == 'Occlusion':
        retrieval_key = 'Occlusion1'
    elif _retrieval_key == 'Resolution':
        retrieval_key = 'Test_Size'
    else:
        retrieval_key = _retrieval_key
    retrieval_mat = retrieval_mat[retrieval_key].squeeze()
    #
    retrieval_dict = collections.defaultdict(lambda : collections.defaultdict(list))
    #
    image_id_set = set()
    #
    for query_idx in tqdm.tqdm(range(len(retrieval_mat["Query"]))):
        #
        query_file = str(retrieval_mat["Query"][query_idx]["imname"][0, 0][0])
        query_bbox = tuple(retrieval_mat["Query"][query_idx]["idlocate"][0, 0][0])
        query_anno_id = anno_lookup[(query_file, query_bbox)]
        #
        query_image_id = image_lookup[query_file]
        #
        image_id_set.add(query_image_id)
        gallery_list = retrieval_mat["Gallery"][query_idx].squeeze()
        for gallery_elem in gallery_list:
            gallery_file = str(gallery_elem[0][0])
            gallery_image_id = image_lookup[gallery_file]
            retrieval_dict['queries'][query_anno_id].append(gallery_image_id)
            # 
            image_id_set.add(gallery_image_id)
    #
    retrieval_dict['images'] = list(image_id_set)
    #
    return dict(retrieval_dict)


# Store datasets in a COCO-like format for easy interaction with COCO primitives
def cuhk2coco(dataset_dir):
    # Create base COCO dict
    coco_dict = {
        'info': {
            'description': 'CUHK-SYSU Person Search Dataset',
            'url': 'http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html',
            'version': '1.0',
            'year': 2016,
            'contributor': 'CUHK-SYSU',
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

    # Retrieval name dict
    retrieval_file_dict = {
        'G50': ('train_test', 'TestG50.mat'),
        'G100': ('train_test', 'TestG100.mat'),
        'G500': ('train_test', 'TestG500.mat'),
        'G1000': ('train_test', 'TestG1000.mat'),
        'G2000': ('train_test', 'TestG2000.mat'),
        'G4000': ('train_test', 'TestG4000.mat'),
        'GOcclusion': ('subset', 'Occlusion.mat'),
        'GResolution': ('subset', 'Resolution.mat'),
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
    partition_dict = get_partition(dataset_dir) 

    # Get CUHK image and annotation metadata
    cuhk_image_dir = os.path.join(dataset_dir, 'Image', 'SSM')
    cuhk_image_dict, cuhk_anno_dict, cuhk_image_lookup, cuhk_anno_lookup = coco_utils.get_coco_dict(partition_dict, cuhk_image_dir)

    # Parse test retrieval scenarios
    retrieval_dir = os.path.join(dataset_dir, 'annotation', 'test')
    for retrieval_name, (retrieval_subdir, retrieval_file) in retrieval_file_dict.items():
        print('==> {}'.format(retrieval_name))
        retrieval_path = os.path.join(retrieval_dir, retrieval_subdir, retrieval_file)
        retrieval_dict = _parse_retrieval(cuhk_image_lookup, cuhk_anno_lookup, retrieval_name, retrieval_path)
        retrieval_json_file = '{}.json'.format(retrieval_name)
        retrieval_json_path = os.path.join(retrieval_json_dir, retrieval_json_file)
        with open(retrieval_json_path, 'w') as fp:
            json.dump(retrieval_dict, fp)

    # Store CUHK metadata in COCO dicts
    trainval_coco_dict['images'] = cuhk_image_dict['train']
    trainval_coco_dict['annotations'] = cuhk_anno_dict['train']
    test_coco_dict['images'] = cuhk_image_dict['test']
    test_coco_dict['annotations'] = cuhk_anno_dict['test']

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
    parser.add_argument('--dataset_dir', type=str, default='/datasets/cuhk')
    args = parser.parse_args() 

    # Build the COCO dataset for CUHK
    cuhk2coco(args.dataset_dir)
