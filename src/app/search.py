# Global imports
import argparse
import torch
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)
import torch.nn.functional as F
## Not used here, but must be loaded for model to work
import torchvision
from tqdm import tqdm
# Libs for data pre-processing
import cv2
import numpy as np
from albumentations.augmentations.geometric import functional as FGeometric
# Libs for loading images
import os
import json
from PIL import Image


# Normalize image tensor using ImageNet stats
def normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.FloatTensor(mean).view(1, 1, 3)
    std = torch.FloatTensor(std).view(1, 1, 3)
    return tensor.div(255.0).sub(mean).div(std)

# Resize image (numpy array) to fit in fixed size window
def window_resize(img, min_size=900, max_size=1500, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    image_min_size = min(width, height)
    image_max_size = max(width, height)
    scale_factor = min_size / image_min_size
    if image_max_size * scale_factor > max_size:
        img_rescaled = FGeometric.longest_max_size(img, max_size=max_size, interpolation=interpolation)
    else:
        img_rescaled = FGeometric.smallest_max_size(img, max_size=min_size, interpolation=interpolation)
    # Compute true scale factor after resize
    true_scale_factor = img_rescaled.shape[0] / img.shape[0]
    # Return scaled image and true scale factor
    return img_rescaled, true_scale_factor

# Convert PIL image to torch tensor
def to_tensor(image, device):
    arr = np.array(image)
    arr_wrs, scale_factor = window_resize(arr)
    tsr = torch.FloatTensor(arr_wrs)
    tsr_norm = normalize(tsr)
    tsr_input = tsr_norm.permute(2, 0, 1).to(device)
    return tsr_input, scale_factor

# Load image to tensor
def load_image(image_path, device):
    image = Image.open(image_path).convert('RGB')
    tensor, scale_factor = to_tensor(image, device)
    return tensor, scale_factor

# Load image dir to tensor
def load_image_dir(image_dir, device):
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        tensor, scale_factor = load_image(image_path, device)
        yield image_file, tensor, scale_factor

# Rescale boxes
def rescale_boxes(boxes, scale_factor):
    return boxes / scale_factor

# Put sample input through model
@torch.no_grad()
def run_model(model, query_path, gallery_dir, device='cuda'):
    # Load query tensor
    query_tsr, qsf = load_image(query_path, device)
    #detections, embeddings, scene_emb = model(images, targets, inference_mode='both')
    # Put query through model
    query_output = model([query_tsr], inference_mode='det')[0]
    # Rescale query boxes
    query_boxes = query_output['det_boxes']
    query_output['det_boxes'] = rescale_boxes(query_boxes, qsf)
    # Load gallery tensors
    gallery_outputs = {}
    for gallery_file, gallery_tsr, gsf in tqdm(load_image_dir(gallery_dir, device)):
        # Put gallery through model
        gallery_output = model([gallery_tsr], inference_mode='det')[0]
        # Rescale gallery boxes
        gallery_boxes = gallery_output['det_boxes']
        gallery_output['det_boxes'] = rescale_boxes(gallery_boxes, gsf)
        # Store outputs
        gallery_outputs[gallery_file] = gallery_output
    # Return query and gallery outputs
    return query_output, gallery_outputs

@torch.no_grad()
def compare_embeddings(model, query_output, gallery_outputs):
    # Get query person embeddings
    query_person_emb = query_output['det_emb']
    query_scene_emb = query_output['scene_emb']

    # For each gallery image
    for gallery_output in gallery_outputs.values():
        ## Get gallery person embeddings
        gallery_person_emb = gallery_output['det_emb']
        gallery_scene_emb = gallery_output['scene_emb']

        ## Compute person similarity: cosine similarity of person embeddings
        person_sim = torch.mm(
            F.normalize(query_person_emb, dim=1),
            F.normalize(gallery_person_emb, dim=1).T
        ).flatten()
        
        ## Store person similarity
        gallery_output['person_sim'] = person_sim

        ## Compute query-scene similarity: cosine similarity of query-gated scene embeddings
        qg_scene_sim = model.gfn.get_scores(query_person_emb, query_scene_emb, gallery_scene_emb).flatten().item()
        
        ## Store query-scene similarity
        gallery_output['gfn_sim'] = qg_scene_sim

# Save results
def save_results(results_dir, results_name, results_dict):
    print('==> Saving results: {}'.format(results_name))
    _results_dir = os.path.join(results_dir, results_name)
    if not os.path.exists(_results_dir):
        os.makedirs(_results_dir)
    for fname, result in results_dict.items():
        name = fname.split('.')[0]
        _results_path = os.path.join(_results_dir, '{}.json'.format(name))
        with open(_results_path, 'w') as fp:
            json.dump({fname:result}, fp)
            print(_results_path)

# Main function
def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--torchscript_path', default='./torchscript/cuhk_final_convnext-base_e30.torchscript.pt')
    parser.add_argument('--query_path', default='./demo/query/chandler_query.jpg')
    parser.add_argument('--gallery_dir', default='./demo/gallery')
    parser.add_argument('--output_dir', default='./demo/output')
    args = parser.parse_args()

    # Get device
    device = torch.device(args.device)

    # Load model
    model = torch.jit.load(args.torchscript_path)

    # Run model
    query_output, gallery_outputs = run_model(
        model, args.query_path, args.gallery_dir, device=device)

    # Compare embeddings
    compare_embeddings(model, query_output, gallery_outputs)

    # Reorganize results
    ## query
    query_file = os.path.basename(args.query_path)
    query_results = {query_file:{
        'det_boxes': query_output['det_boxes'].cpu().tolist(),
        'det_scores': query_output['det_scores'].cpu().tolist(),
    }}
    ## gallery
    gallery_results = {gallery_file:{
        'det_boxes': gallery_output['det_boxes'].cpu().tolist(),
        'det_scores': gallery_output['det_scores'].cpu().tolist(),
        'person_sim': gallery_output['person_sim'].cpu().tolist(),
        'gfn_sim': gallery_output['gfn_sim'],
    } for gallery_file, gallery_output in gallery_outputs.items()}

    # Save results
    save_results(args.output_dir, 'query', query_results)
    save_results(args.output_dir, 'gallery', gallery_results)


# Run this module
if __name__ == '__main__':
    # Run main
    main()
