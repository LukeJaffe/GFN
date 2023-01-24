# Global imports
import os
import collections
import json
import numpy as np
from PIL import Image
import tqdm


# Get image and annotation metadata for dataset in COCO format
def get_coco_dict(partition_dict, image_dir, cam_id_dict=None):
    # Dicts to store results
    anno_dict = collections.defaultdict(list)
    image_dict = collections.defaultdict(list)
    anno_lookup = {}
    image_lookup = {}

    # Build COCO annotation dict
    image_id, anno_id = 0, 0
    for mode in partition_dict:
        num_boxes = 0
        for image_file, data_list in partition_dict[mode].items():
            # Load image
            image_path = os.path.join(image_dir, image_file)

            # Get image info
            w, h = Image.open(image_path).size
            ## Optional cam_id
            if cam_id_dict is not None:
                cam_id = cam_id_dict[image_file]
            else:
                cam_id = -1

            # Store image info with a unique ID
            image_dict[mode].append({
                'file_name': image_file,
                'height': h,
                'width': w,
                'id': image_id,
                'cam_id': cam_id,
            })
            image_lookup[image_file] = image_id

            # Store anno info
            num_boxes += len(data_list)
            for bbox, pid, is_known in data_list:
                # Pre-compute the IoU threshold specified by the original CUHK-SYSU dataset paper
                _bbox = [float(x) for x in bbox]
                bw, bh = _bbox[2:]
                iou_thresh = min(0.5, (bw * bh * 1.0) / ((bw + 10) * (bh + 10)))
                # Store this annotation with a unique ID
                anno_dict[mode].append({
                    'area': _bbox[2]*_bbox[3],
                    'image_id': image_id,
                    'bbox': _bbox,
                    'id': anno_id,
                    'person_id': pid,
                    'is_known': is_known,
                    'iou_thresh': iou_thresh,
                    'category_id': 1,
                    'iscrowd': 0,
                })
                anno_lookup[(image_file, tuple(bbox))] = anno_id
                anno_id += 1
            image_id += 1

        print('{} num images: {}'.format(mode, len(image_dict[mode])))
        print('{} num boxes: {}'.format(mode, num_boxes))

    return image_dict, anno_dict, image_lookup, anno_lookup


# Build a retrieval partition from a coco-format dict with a fixed gallery size
def build_coco_retrieval(retrieval_dir, retrieval_name, coco_dict, gallery_size=100, seed=0):
    # Fix the random seed
    np.random.seed(seed)
    person_id_dict = collections.defaultdict(list)
    for anno in coco_dict['annotations']:
        person_id_dict[anno['person_id']].append(anno)  
    image_id_list = [image['id'] for image in coco_dict['images']]
    retrieval_dict = collections.defaultdict(lambda : collections.defaultdict(list))
    full_image_id_list = []
    for person_id, anno_list in tqdm.tqdm(person_id_dict.items()):
        if len(anno_list) > 1:
            for query_anno in anno_list:
                query_id = query_anno['id']
                query_image_id = query_anno['image_id']
                full_image_id_list.append(query_image_id)
                # First, add all images with a query match to the gallery
                for gallery_anno in anno_list:
                    gallery_id = gallery_anno['id']
                    gallery_image_id = gallery_anno['image_id']
                    if query_id != gallery_id:
                        if len(retrieval_dict['queries'][query_id]) < gallery_size:
                            retrieval_dict['queries'][query_id].append(gallery_image_id)
                # Then, pad the gallery to gallery_size with other randomly selected test images
                np.random.shuffle(image_id_list)
                for gallery_image_id in image_id_list:
                    if len(retrieval_dict['queries'][query_id]) == gallery_size:
                        break
                    if gallery_image_id not in retrieval_dict['queries'][query_id]:
                        retrieval_dict['queries'][query_id].append(gallery_image_id)

    # Check validity of the retrieval set, and get list of all images in the set
    for query_id, gallery_image_id_list in retrieval_dict['queries'].items():
        assert len(gallery_image_id_list) == gallery_size
        full_image_id_list.extend(gallery_image_id_list) 
    full_image_id_list = list(set(full_image_id_list))
    retrieval_dict['images'] = full_image_id_list

    # Report set info
    print('num queries: {}/{}'.format(len(retrieval_dict['queries']), len(coco_dict['annotations'])))
    print('num images: {}/{}'.format(len(retrieval_dict['images']), len(coco_dict['images'])))

    # Save retrieval data to disk
    retrieval_path = os.path.join(retrieval_dir, '{}.json'.format(retrieval_name))
    with open(retrieval_path, 'w') as fp:
        json.dump(retrieval_dict, fp)
