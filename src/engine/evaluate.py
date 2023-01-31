# Global imports
import collections
import torch
import torch.nn.functional as F
import os
import json
import numpy as np
from pprint import pprint
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from ray.tune.integration.torch import is_distributed_trainable
from torchvision.ops import boxes as box_ops
# Package imports
from osr.engine import utils as engine_utils


# Define objects to neatly store results
QueryLookupEntry = collections.namedtuple('QueryLookupEntry',
    ['image_id', 'person_id', 'embedding', 'box', 'idx'],
    defaults=[None, None, None, None, None],
)

ImageLookupEntry = collections.namedtuple('ImageLookupEntry',
    ['id', 'person_ids', 'iou_thresh', 'boxes', 'scores', 'embeddings', 'features'],
)

DetectionLookupEntry = collections.namedtuple('DetectionLookupEntry',
    ['boxes', 'scores', 'embeddings', 'labels'],
)

RetrievalLookupEntry = collections.namedtuple('RetrievalLookupEntry',
    ['sims', 'boxes', 'iou', 'gfn_scores'],
    defaults=[None, None, None, None],
)

Protocol = collections.namedtuple('Protocol',
    ['partition_name', 'name', 'data']
)

RetrievalBox = collections.namedtuple('RetrievalBox',
    ['image_id', 'box', 'score', 'match', 'query'],
    defaults=[None, None, None, None, None],
)



# Get retrieval protocol information
def get_protocol_list(data_loader):
    partition_name = data_loader.sampler.partition_name
    full_query_id_list = data_loader.sampler.query_id_list
    retrieval_dir = data_loader.sampler.retrieval_dir
    retrieval_name_list = data_loader.sampler.retrieval_name_list
    _retrieval_name_list = [n for n in retrieval_name_list if n != 'all']
    protocol_list = []
    for retrieval_name in _retrieval_name_list:
        retrieval_path = os.path.join(retrieval_dir, '{}.json'.format(retrieval_name))
        with open(retrieval_path, 'r') as fp:
            _retrieval_dict = json.load(fp)
            protocol_list.append(Protocol(partition_name=partition_name,
                name=retrieval_name, data=_retrieval_dict))
    if 'all' in retrieval_name_list:
        protocol_list.append(Protocol(partition_name=partition_name,
            name='all', data=full_query_id_list))
    return protocol_list


def get_query_embeddings(query_lookup, image_lookup):
    # Combine all query embeddings into a single tensor for easy computation of cosine similarity
    query_embedding_list = []
    query_image_feat_list = []
    for _id_idx, (_id, _query) in enumerate(query_lookup.items()):
        _embedding = _query.embedding
        query_embedding_list.append(_embedding)
        query_lookup[_id] = query_lookup[_id]._replace(idx=_id_idx)
        query_image_id = query_lookup[_id].image_id
        query_image_feat = image_lookup[query_image_id].features
        query_image_feat_list.append(query_image_feat)
    query_embeddings = torch.cat(query_embedding_list, dim=0)
    return query_embeddings, query_image_feat_list


def get_gfn_scores(model, image_lookup, query_embeddings, query_image_feat_list, use_gfn=False, device='cuda'):
    # Compute GFN scores
    if use_gfn:
        ## Pack feats together for GFN or QFN
        query_img_feat_mat = torch.cat(query_image_feat_list, dim=0)
        gallery_img_feat_mat = torch.cat([image_lookup[query_id].features for query_id in image_lookup], dim=0)
        ## Get GFN
        if is_distributed_trainable():
            gfn = model.module.gfn
        else:
            gfn = model.gfn
        ## Get scores
        gfn_scores = gfn.get_scores(query_embeddings, query_img_feat_mat, gallery_img_feat_mat)
        print(gfn_scores.min(), gfn_scores.max())
        gfn_score_dict = dict(zip(list(image_lookup.keys()), gfn_scores.T.to(device)))
    else:
        gfn_score_dict = None
    return gfn_score_dict


def run_model(model, data_loader, use_amp=False, device='cuda'):
    # Compute and gather detections and ground truth embeddings
    print('==> Computing detections and embeddings')
    with torch.cuda.amp.autocast(enabled=use_amp):
        image_lookup = {}
        query_lookup = {}
        detection_lookup = {}
        for iter_idx, (images, targets) in tqdm(enumerate(data_loader), ncols=0, total=len(data_loader)):
            images, targets = engine_utils.to_device(images, targets, device)
            outputs = model(images, targets, inference_mode='both')
            for output in outputs:
                print(output['gt_emb'].shape)
            embeddings = torch.cat([output['gt_emb'] for output in outputs])
            assert len(targets) == len(outputs)
            # XXX
            for target, output in zip(targets, outputs):
                image_id = target['image_id'].item()
                image_lookup[image_id] = ImageLookupEntry(
                    id=target['id'], person_ids=target['person_id'], iou_thresh=target['iou_thresh'].cpu(),
                    boxes=target['boxes'], scores=torch.ones(target['boxes'].size(0)), embeddings=embeddings,
                    features=output['scene_emb'],
                )
                detection_lookup[image_id] = DetectionLookupEntry(
                    boxes=output['det_boxes'],
                    scores=output['det_scores'],
                    labels=output['det_labels'],
                    embeddings=output['det_emb'],
                )
                #
                assert len(target['id']) == len(target['person_id']) == len(embeddings)
                for _id, _person_id, _box, _embedding in zip(target['id'].tolist(), target['person_id'], target['boxes'], embeddings.unsqueeze(1)):
                    if _id in data_loader.sampler.query_id_list:
                        query_lookup[_id] = QueryLookupEntry(image_id=image_id, person_id=_person_id,
                            embedding=_embedding, box=_box)
    #
    return query_lookup, image_lookup, detection_lookup


@torch.no_grad()
def get_model_output(model, data_loader, use_gfn=False, use_amp=False, device='cuda'):
    # Put model in eval mode
    model.eval()

    # Compute and gather detections and ground truth embeddings
    query_lookup, image_lookup, detection_lookup = run_model(model, data_loader,
        use_amp=use_amp, device=device)

    # Combine all query embeddings into a single tensor for easy computation of cosine similarity
    query_embeddings, query_image_feat_list = get_query_embeddings(query_lookup, image_lookup)

    # Compute GFN scores
    gfn_score_dict = get_gfn_scores(model, image_lookup, query_embeddings, query_image_feat_list,
        use_gfn=use_gfn, device=device)

    # Return results
    return query_lookup, query_embeddings, image_lookup, detection_lookup, gfn_score_dict


@torch.no_grad()
def evaluate_performance(
    model, data_loader, device,
    use_amp=False, use_gfn=False, gfn_mode=None,
    report_timing=False,
):
    # Get model output
    query_lookup, query_embeddings, image_lookup, detection_lookup, gfn_score_dict = get_model_output(model, data_loader,
        use_gfn=use_gfn, use_amp=use_amp, device=device)

    # Dicts to store results
    metric_dict, value_dict = {}, {}

    # Compute detection performance
    print('==> Computing detection performance')
    detection_metric_dict, retrieval_dict = evaluate_detection_orig(data_loader.sampler.partition_name,
        detection_lookup, image_lookup, query_embeddings, gfn_score_dict=gfn_score_dict)
    gt_detection_metric_dict, gt_retrieval_dict = evaluate_detection_orig(data_loader.sampler.partition_name,
        image_lookup, image_lookup, query_embeddings, gfn_score_dict=gfn_score_dict)
    metric_dict.update(detection_metric_dict)
    print('det:')
    pprint(detection_metric_dict)
    print('gt:')
    pprint(gt_detection_metric_dict)

    # Get retrieval protocol information
    protocol_list = get_protocol_list(data_loader)

    # Compute retrieval performance
    print('==> Computing retrieval performance (protocol)')
    for protocol in protocol_list:
        print('==> Protocol: {}'.format(protocol.name))
        for _use_gfn in [False, True] if use_gfn else [False]:
            retrieval_metric_dict, retrieval_value_dict = evaluate_retrieval_orig(protocol,
                retrieval_dict, query_lookup, image_lookup, use_gt=False, use_gfn=_use_gfn)
            gt_retrieval_metric_dict, gt_retrieval_value_dict = evaluate_retrieval_orig(protocol,
                gt_retrieval_dict, query_lookup, image_lookup, use_gt=True, use_gfn=_use_gfn)
            print('det:')
            pprint(retrieval_metric_dict)
            print('gt:')
            pprint(gt_retrieval_metric_dict)
            # Store results for this set
            metric_dict.update(gt_retrieval_metric_dict)
            metric_dict.update(retrieval_metric_dict)
            value_dict.update(gt_retrieval_value_dict)
            value_dict.update(retrieval_value_dict)

    # Report metrics 
    print('\n==> Full metrics:')

    # Return results
    return metric_dict, value_dict


# Function to match predicted boxes to ground truth boxes
def _match_boxes(match_quality_matrix, iou_thresh):
    """
    Note: This function mimics other person search codes,
    and does not implement the standard COCO box matching
    algorithm.
    """
    # Zero out any pairs with IoU < thresh
    match_quality_matrix[match_quality_matrix<iou_thresh] = 0.0
    # For each det, keep largest IoU of all GT
    max_gt_idx = match_quality_matrix.argmax(dim=1, keepdim=True)
    max_gt_mask = torch.zeros_like(match_quality_matrix, dtype=torch.bool)
    max_gt_mask.scatter_(1, max_gt_idx, True)
    # For each GT, keep largest IoU of all det
    max_det_idx = match_quality_matrix.argmax(dim=0, keepdim=True)
    max_det_mask = torch.zeros_like(match_quality_matrix, dtype=torch.bool)
    max_det_mask.scatter_(0, max_det_idx, True)
    ## Set non-max elems to 0
    match_quality_matrix[~max_gt_mask] = 0.0
    match_quality_matrix[~max_det_mask] = 0.0
    # Get indices
    det_idx, gt_idx = torch.where(match_quality_matrix)
    return det_idx, gt_idx


# Detection evaluation function
def evaluate_detection_orig(partition_name, detection_lookup, image_lookup, query_embeddings,
        gfn_score_dict=None, det_thresh=0.5, iou_thresh=0.5):
    num_gt_match, num_gt_tot = 0, 0
    num_det_tot = 0
    det_scores_list = []
    det_matches_list = []
    retrieval_lookup = {}
    for image_id in tqdm(image_lookup):
        # Unpack detections for this image
        detection = detection_lookup[image_id]
        det_boxes = detection.boxes
        det_scores = detection.scores
        det_embeddings = detection.embeddings
        num_det_tot += det_boxes.shape[0]
        # Unpack GT for this image
        gt = image_lookup[image_id]
        gt_boxes = gt.boxes
        gt_person_ids = gt.person_ids
        num_gt_tot += gt_boxes.shape[0]
        # Store GFN scores
        if gfn_score_dict is not None:
            retrieval_lookup[image_id] = RetrievalLookupEntry(gfn_scores=gfn_score_dict[image_id])
        else:
            retrieval_lookup[image_id] = RetrievalLookupEntry()

        # Compute detection results
        if det_boxes.shape[0] > 0:
            # Filter only detections with high enough score
            det_mask = det_scores >= det_thresh
            good_det_scores = det_scores[det_mask]
            good_det_boxes = det_boxes[det_mask]
            good_det_embeddings = det_embeddings[det_mask]
            # Match detections with GT boxes
            match_quality_matrix = box_ops.box_iou(good_det_boxes, gt_boxes)
            det_idx, gt_idx = _match_boxes(match_quality_matrix.clone(), iou_thresh)
            #
            num_gt_match += gt_idx.shape[0]
            det_scores_list.append(good_det_scores)
            det_matches = torch.zeros(len(good_det_boxes))
            det_matches[det_idx] = 1
            det_matches_list.append(det_matches)
            # Store detected person_ids
            det_person_ids = torch.full((len(good_det_boxes),), -1, dtype=torch.long)
            det_person_ids[det_idx] = gt_person_ids[gt_idx]
            # Compute cosine similarity used later for retrieval ranking
            ## We assume the embeddings are already normalized, and have other scores incorporated after normalization:
            ## confidence-weighted similarity from the detector
            det_sims = torch.mm(
                query_embeddings,
                good_det_embeddings.T,
            )
            # Store everything in retrieval dict
            assert det_matches.shape[0] == det_person_ids.shape[0] == det_sims.shape[1]
            retrieval_lookup[image_id] = retrieval_lookup[image_id]._replace(sims=det_sims, boxes=good_det_boxes, iou=match_quality_matrix)
    #
    print('num_det_tot:', num_det_tot)
    print('gt/tot:', num_gt_match, num_gt_tot)
    det_recall = num_gt_match / num_gt_tot
    # Compute AP@0.5
    if len(det_scores_list) > 0:
        ## Combine all scores, labels, and clean any invalid scores
        det_scores = np.nan_to_num(torch.cat(det_scores_list).tolist(), posinf=0, neginf=0)
        det_matches = torch.cat(det_matches_list).tolist()
        ## Compute AP@0.5
        det_ap = average_precision_score(det_matches, det_scores) * det_recall
    else:
        det_ap = 0
    metric_dict = {
        f'{partition_name}_ap@0.5': det_ap,
        f'{partition_name}_recall@0.5': det_recall,
    }
    #
    return metric_dict, retrieval_lookup


# Person search retrieval evaluation function
def evaluate_retrieval_orig(protocol,
        retrieval_lookup, query_lookup, image_lookup,
        iou_thresh=0.5, iou_thresh_mode='variable', use_gt=False, use_gfn=False):
    """
    Function for person search retrieval evaluation on arbitrary datasets.
    Designed and tested for CUHK-SYSU and PRW datasets, but extensible to
    other datasets following the same format.

    This function was designed to exactly replicate the erroneous behavior
    in the original person search eval code, used by JDIFL, NAE, SeqNet,
    and others. The difference in performance from this erroneous evaluation
    is small, but important to replicate for exact fair comparison of
    results between different methods.
    
    The corrected function is actually much shorter since it doesn't need
    to handle as many corner cases.
    """

    # Variables to store results
    top1_list, ap_list = [], []
    gfn_top1_list, gfn_ap_list = [], []
    full_gfn_match_list, full_gfn_score_list = [], []
    tot_pred_match_count, tot_gt_match_count = 0, 0
    tot_pred_match1_count, tot_gt_match1_count = 0, 0
    num_no_gt = 0

    # Unpack protocol data
    if protocol.name == 'all':
        query_id_list = protocol.data
        protocol_dict = {'queries': None}
    else:
        protocol_dict = protocol.data
        query_id_list = [int(x) for x in protocol_dict['queries']]

    # Iterate through each query
    for query_iter, query_id in tqdm(list(enumerate(query_id_list))):
        # Get query data
        query_idx = query_lookup[query_id].idx
        query_person_id = query_lookup[query_id].person_id
        query_image_id = query_lookup[query_id].image_id

        # Storage variables for this query
        sim_list = []
        gallery_person_matches_list = []
        gfn_score_list, gfn_match_list = [], []
        gt_match_count = 0

        # Set gallery image ids for this query based on the protocol
        if protocol.name == 'all':
            gallery_image_ids = retrieval_lookup.keys()
        else:
            if type(protocol_dict['queries']) == list:
                gallery_image_ids = protocol_dict['images']
            elif type(protocol_dict['queries']) == dict:
                gallery_image_ids = protocol_dict['queries'][str(query_id)]

        # Iterate through each gallery image for this query
        gallery_image_set = set()
        for gallery_image_id in gallery_image_ids:
            # Skip the identity search
            if type(protocol_dict['queries']) == list:
                if query_image_id == gallery_image_id:
                    continue

            # Get the detections for this image
            detection = retrieval_lookup[gallery_image_id]

            # Count ground truth matches to the query person in this gallery image
            gt_person_ids = image_lookup[gallery_image_id].person_ids
            gt_person_matches = query_person_id == gt_person_ids
            _gt_match_count = gt_person_matches.sum().item()

            # Increment the total number of GT matches for this query
            ## Add 1, not _gt_match_count, to handle corner cases where _gt_match_count > 1
            gt_match_count += _gt_match_count > 0

            # If we are using the Gallery Filter Network
            if use_gfn:
                gallery_gfn_scores = detection.gfn_scores[query_idx]
                # Track GFN stats
                gfn_score_list.append(gallery_gfn_scores.item())
                gfn_match_list.append(float(_gt_match_count > 0))

            # If there are no detects, continue to next gallery image
            if detection.sims is None:
                continue

            # Handle mistake from CUHK dataset: repeated image in the gallery
            if gallery_image_id in gallery_image_set:
                # Intentionally skip evaluating this prediction if it is a duplicate gallery image
                ## Yes, even after counting GTs (we know this is erroneous: mimics other code behavior)
                continue
            else:
                gallery_image_set.add(gallery_image_id)

            # Get sims for this query
            gallery_sims = detection.sims[query_idx]

            # If using GFN, multiply sims by GFN scores
            if use_gfn:
                gallery_sims = gallery_sims * gallery_gfn_scores
            gallery_iou = detection.iou
            _gallery_person_matches = torch.zeros(gallery_sims.shape[0], dtype=torch.bool)

            # If there is at least one ground truth match for the query in this gallery image
            if _gt_match_count > 0:
                # Get match indices
                gt_idx = torch.where(gt_person_matches)[0]
                _gallery_iou = gallery_iou[:, gt_idx] 

                # Handle mistake from CUHK dataset: repeated person_id in a gallery image
                if _gt_match_count > 1:
                    ## Make sure there is only 1 repeat and not more
                    assert _gt_match_count == 2
                    ## Use the first box by default, which mimics previous CUHK eval by design
                    _gallery_iou = _gallery_iou[:, 0].unsqueeze(1)

                # Get indices where IoU is above the required threshold
                ## Using a fixed IoU threshold
                if iou_thresh_mode == 'fixed':
                    _iou_idx = torch.where(_gallery_iou >= iou_thresh)[0]
                ## Using the variable size-based threshold (standard)
                elif iou_thresh_mode == 'variable':
                    var_iou_thresh = image_lookup[gallery_image_id].iou_thresh[gt_person_matches][0].item()
                    _iou_idx = torch.where(_gallery_iou >= var_iou_thresh)[0]

                # Mark matching positions in the gallery
                if len(_iou_idx) > 0:
                    _gallery_sims = gallery_sims[_iou_idx]
                    _sim_idx = torch.argsort(_gallery_sims, descending=True)
                    _match_iou_idx = _iou_idx[_sim_idx[0]]
                    _gallery_person_matches[_match_iou_idx] = True

            # Store resulting matches and sims
            assert _gallery_person_matches.sum() <= 1
            gallery_person_matches_list.append(_gallery_person_matches)
            sim_list.append(gallery_sims)

        # Compute metrics if there is at least one prediction
        if len(sim_list) > 0:
            gallery_sims = torch.cat(sim_list)
            gallery_person_matches = torch.cat(gallery_person_matches_list)
            gallery_matches = gallery_person_matches.float()
            # Compute recall for this query
            pred_match_count = gallery_person_matches.sum().item()
            tot_pred_match_count += pred_match_count
            tot_pred_match1_count += 0 if pred_match_count == 0 else 1
            # Corner case: no GT matches for this query
            if gt_match_count == 0:
                num_no_gt += 1
                ap = 0
                top1 = 0
            else:
                query_recall = pred_match_count / gt_match_count
                top1_idx = gallery_sims.argmax()
                top1 = gallery_matches[top1_idx].item()
                if gallery_matches.sum().item() > 0:
                    ## Compute AP on cleaned input
                    ap = average_precision_score(gallery_matches.tolist(),
                        np.nan_to_num(gallery_sims.tolist(), posinf=0, neginf=0)
                    ) * query_recall
                else:
                    ap = 0
        else:
            top1, ap = 0, 0

        # Store retrieval metrics for this query
        tot_gt_match_count += gt_match_count
        tot_gt_match1_count += 1
        ap_list.append(ap)
        top1_list.append(top1)

        # If we are using the Gallery Filter Network
        if use_gfn:
            ## GFN AP
            if sum(gfn_match_list) > 0:
                ### Compute AP on cleaned input
                gfn_ap = average_precision_score(gfn_match_list, np.nan_to_num(gfn_score_list, posinf=0, neginf=0))
            else:
                gfn_ap = 0
            ## GFN top-1 accuracy
            if (len(gfn_score_list) > 0) and (len(gfn_match_list) > 0):
                gfn_top_idx = np.argmax(gfn_score_list)
                gfn_top1 = gfn_match_list[gfn_top_idx]
            else:
                gfn_top1 = 0
            ## GFN neg filter @recall=0.99
            full_gfn_match_list.extend(gfn_match_list)
            full_gfn_score_list.extend(gfn_score_list) 
            ## Store metrics
            gfn_ap_list.append(gfn_ap)
            gfn_top1_list.append(gfn_top1)

    # Compute final summary metrics
    print('match: {}/{}'.format(tot_pred_match1_count, tot_gt_match1_count))
    print('recall: {}/{}'.format(tot_pred_match_count, tot_gt_match_count))
    print('num no gt: {}'.format(num_no_gt))
    match = tot_pred_match1_count / tot_gt_match1_count
    recall = tot_pred_match_count / tot_gt_match_count
    top1 = np.mean(top1_list)

    # Show indices of wrong top1 pred
    top1_tsr = torch.FloatTensor(top1_list)
    top1_idx = torch.where(top1_tsr==0)[0]
    mAP = np.mean(ap_list)

    # Store metrics
    ## prefix strings
    source_name = 'gt' if use_gt else 'det'
    gfn_name = '_gfn' if use_gfn else ''
    metric_dict = {
        f'{protocol.partition_name}_{protocol.name}_{source_name}{gfn_name}_match': match,
        f'{protocol.partition_name}_{protocol.name}_{source_name}{gfn_name}_recall': recall,
        f'{protocol.partition_name}_{protocol.name}_{source_name}{gfn_name}_mAP': mAP,
        f'{protocol.partition_name}_{protocol.name}_{source_name}{gfn_name}_top1': top1,
    }
    value_dict = {
        f'{protocol.partition_name}_{protocol.name}_{source_name}{gfn_name}_top1_list': top1_list,
    }

    # Compute GFN summary metrics
    if use_gfn:
        ## GFN retrieval stats
        gfn_mAP = np.mean(gfn_ap_list)
        gfn_top1 = np.mean(gfn_top1_list)

        ## GFN neg filter (NPV) @recall=0.99
        full_gfn_match_arr = np.array(full_gfn_match_list)
        full_gfn_score_arr = np.array(full_gfn_score_list)
        sort_idx = np.argsort(-full_gfn_score_arr)
        sorted_gfn_match_arr = full_gfn_match_arr[sort_idx]
        sorted_gfn_score_arr = full_gfn_score_arr[sort_idx]
        gfn_pos_mask = sorted_gfn_match_arr == 1
        pos_gfn_score_arr = sorted_gfn_score_arr[gfn_pos_mask]
        num_pos = gfn_pos_mask.sum().item()
        num_pos99 = int(num_pos * 0.99)
        gfn_filter_thresh = pos_gfn_score_arr[num_pos99].item()
        gfn_neg_mask = ~gfn_pos_mask
        neg_gfn_score_arr = sorted_gfn_score_arr[gfn_neg_mask]
        num_gfn_filter = (neg_gfn_score_arr < gfn_filter_thresh).sum().item()
        num_neg = gfn_neg_mask.sum().item()
        if num_neg > 0:
            gfn_frac_neg_filter = num_gfn_filter / num_neg
        else:
            gfn_frac_neg_filter = 0.0

        ## Store full gfn scores
        value_dict.update({
            f'{protocol.partition_name}_{protocol.name}_image_gfn_match_list': full_gfn_match_list,
            f'{protocol.partition_name}_{protocol.name}_image_gfn_score_list': full_gfn_score_list,
        })

        ## Store metrics
        metric_dict.update({
            f'{protocol.partition_name}_{protocol.name}_image_gfn_mAP': gfn_mAP,
            f'{protocol.partition_name}_{protocol.name}_image_gfn_top1': gfn_top1,
            f'{protocol.partition_name}_{protocol.name}_image_gfn_frac_filter': gfn_frac_neg_filter,
            f'{protocol.partition_name}_{protocol.name}_image_gfn_thresh_filter': gfn_filter_thresh,
        })

    # Return metrics and other values
    return metric_dict, value_dict


# Function to get retrieval results for displaying
def get_retrieval_results(protocol,
        retrieval_lookup, query_lookup, image_lookup, subset_idx=None, exclude_self=True,
        iou_thresh=0.5, iou_thresh_mode='fixed', use_gt=False, use_gfn=False):
    """
    Mostly the same as "evaluate_retrieval_orig", but with less metric tabulation.
    Could use some cleanup, since variables used for metrics are still here, but
    not used.
    """

    # Variables to store results
    top1_list, ap_list = [], []
    gfn_top1_list, gfn_ap_list = [], []
    full_gfn_match_list, full_gfn_score_list = [], []
    tot_pred_match_count, tot_gt_match_count = 0, 0
    tot_pred_match1_count, tot_gt_match1_count = 0, 0
    num_no_gt = 0

    # Unpack protocol data
    if protocol.name == 'all':
        query_id_list = protocol.data
        protocol_dict = {'queries': None}
    else:
        protocol_dict = protocol.data
        query_id_list = [int(x) for x in protocol_dict['queries']]

    # Use specific subset of query ids
    if subset_idx is not None:
        query_id_list = [query_id_list[i] for i in subset_idx]

    # 
    result_list = []

    # Iterate through each query
    for query_iter, query_id in tqdm(list(enumerate(query_id_list))):
        query_idx = query_lookup[query_id].idx
        query_person_id = query_lookup[query_id].person_id
        query_image_id = query_lookup[query_id].image_id
        sim_list = []
        image_id_list = []
        box_list = []
        gallery_person_matches_list = []
        gfn_score_list, gfn_match_list = [], []
        gt_match_count = 0

        # Set gallery image ids for this query based on the protocol
        if protocol.name == 'all':
            gallery_image_ids = retrieval_lookup.keys()
        else:
            if type(protocol_dict['queries']) == list:
                gallery_image_ids = protocol_dict['images']
            elif type(protocol_dict['queries']) == dict:
                gallery_image_ids = protocol_dict['queries'][str(query_id)]

        # Iterate through each gallery image for this query
        gallery_image_set = set()
        for gallery_image_id in gallery_image_ids:
            # Skip the identity search
            if (type(protocol_dict['queries']) == list) or exclude_self:
                if query_image_id == gallery_image_id:
                    continue

            # Get the detections for this image
            detection = retrieval_lookup[gallery_image_id]

            # Count ground truth matches to the query person in this gallery image
            gt_person_ids = image_lookup[gallery_image_id].person_ids
            gt_person_matches = query_person_id == gt_person_ids
            _gt_match_count = gt_person_matches.sum().item()

            # Increment the total number of GT matches for this query
            ## Add 1, not _gt_match_count, to handle corner cases where _gt_match_count > 1
            gt_match_count += _gt_match_count > 0

            # If we are using the Gallery Filter Network
            if use_gfn:
                gallery_gfn_scores = detection.gfn_scores[query_idx]
                # Track GFN stats
                gfn_score_list.append(gallery_gfn_scores.item())
                gfn_match_list.append(float(_gt_match_count > 0))

            # If there are no detects, continue to next gallery image
            if detection.sims is None:
                continue

            # Handle mistake from CUHK dataset: repeated image in the gallery
            if gallery_image_id in gallery_image_set:
                # Intentionally skip evaluating this prediction if it is a duplicate gallery image
                ## Yes, even after counting GTs (we know this is wrong)
                continue
            else:
                gallery_image_set.add(gallery_image_id)

            # Get sims for this query
            gallery_sims = detection.sims[query_idx]
            gallery_boxes = detection.boxes

            # If using GFN, multiply sims by GFN scores
            if use_gfn:
                gallery_sims = gallery_sims * gallery_gfn_scores
            gallery_iou = detection.iou
            _gallery_person_matches = torch.zeros(gallery_sims.shape[0], dtype=torch.bool)

            # If there is at least one ground truth match for the query in this gallery image
            if _gt_match_count > 0:
                #
                gt_idx = torch.where(gt_person_matches)[0]
                _gallery_iou = gallery_iou[:, gt_idx] 

                # Handle mistake from CUHK dataset: repeated person_id in a gallery image
                if _gt_match_count > 1:
                    ## Make sure there is only 1 repeat and not more
                    assert _gt_match_count == 2
                    ## Use the first box by default, which mimics previous CUHK eval by design
                    _gallery_iou = _gallery_iou[:, 0].unsqueeze(1)

                #
                if iou_thresh_mode == 'fixed':
                    _iou_idx = torch.where(_gallery_iou >= iou_thresh)[0]
                elif iou_thresh_mode == 'variable':
                    var_iou_thresh = image_lookup[gallery_image_id].iou_thresh[gt_person_matches][0].item()
                    _iou_idx = torch.where(_gallery_iou >= var_iou_thresh)[0]
                #
                if len(_iou_idx) > 0:
                    _gallery_sims = gallery_sims[_iou_idx]
                    _sim_idx = torch.argsort(_gallery_sims, descending=True)
                    _match_iou_idx = _iou_idx[_sim_idx[0]]
                    _gallery_person_matches[_match_iou_idx] = True
            assert _gallery_person_matches.sum() <= 1
            gallery_person_matches_list.append(_gallery_person_matches)
            #
            sim_list.append(gallery_sims)
            image_id_list.extend([gallery_image_id]*len(gallery_sims))
            box_list.append(gallery_boxes)
        #
        if len(sim_list) > 0:
            gallery_sims = torch.cat(sim_list)
            gallery_boxes = torch.cat(box_list)
            gallery_image_ids = torch.LongTensor(image_id_list)
            gallery_person_matches = torch.cat(gallery_person_matches_list)
            gallery_matches = gallery_person_matches.float()
            assert gallery_sims.shape[0] == gallery_boxes.shape[0] == gallery_image_ids.shape[0]
            # Corner case: no GT matches for this query
            if gt_match_count == 0:
                print('No matches...')
                result_list.append(None)
            else:
                top1_idx = gallery_sims.argmax().item()
                match = gallery_matches[top1_idx].item()
                top_gallery_box = RetrievalBox(image_id=gallery_image_ids[top1_idx].item(),
                    box=gallery_boxes[top1_idx],
                    score=gallery_sims[top1_idx],
                    match=match,
                    query=False,
                )    
                top_query_box = RetrievalBox(image_id=query_image_id,
                    box=query_lookup[query_id].box,
                    query=True,
                )
                result_list.append((top_query_box, top_gallery_box))
        else:
            print('No detects...')
            result_list.append(None)
    #
    return result_list
