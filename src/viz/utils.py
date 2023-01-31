# Global imports
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from IPython.display import HTML


# Plot detected boxes on image with matplotlib
def show_detects(image, detect,
                 show_det_score=False, show_sim_score=False,
                 ax=None, title=None, xlabel=None, figsize=(12, 12)):
    # Setup subplot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Setup labels
    if title is not None:
        ax.set_title(title, fontsize=20, fontweight='bold')
    if 'gfn_sim' in detect:
        ax.set_xlabel('GFN Score: {:.2f}'.format(detect['gfn_sim']), fontsize=20)
    # Show the image
    ax.imshow(image)
    # Plot boxes (and optionally similarity scores)
    for i, (box, score) in enumerate(zip(detect['det_boxes'], detect['det_scores'])):
        x, y, x2, y2 = box
        w, h = x2 - x, y2 - y
        ax.add_patch(Rectangle((x, y), w, h, edgecolor='green', lw=4, fill=False, alpha=0.8))
        ax.add_patch(Rectangle((x+2, y+2), w-4, h-4, edgecolor='whitesmoke', lw=1, fill=False, alpha=0.8))
        ##
        label = ''
        ## Display detected box scores
        if show_det_score:
            label += 'd={:.2f}'.format(detect['det_scores'][i])
        ## Display person similarity
        if show_sim_score:
            if len(label) > 0:
                label += ' / '
            label += 's={:.2f}'.format(detect['person_sim'][i])
        if len(label) > 0:
            ax.text(x, y, label, ha="left", va="bottom", size=14,
                bbox=dict(boxstyle="square,pad=0.2", fc="whitesmoke", alpha=0.8, ec='black', lw=2.0)
            )
    # Remove ticks and expand borders
    ax.set_xticks([])
    ax.set_yticks([])
    [x.set_linewidth(3) for x in ax.spines.values()]

# Return list of detected (PIL) image crops
def get_crops(image, detect, ax=None):
    # Extract crops using detected boxes
    crop_list = []
    for i, box in enumerate(detect['det_boxes']):
        x1, y1, x2, y2 = box
        crop = image.crop((x1, y1, x2, y2))
        crop_list.append(crop)
    return crop_list

# Show ranked re-id crops
def show_reid(crop_list, score_list, plot_width=1, show_topk=10):
    # Sort crops by decreasing score
    sorted_score_idx = np.argsort(score_list)[::-1]
    sorted_score_list = [score_list[i] for i in sorted_score_idx][:show_topk]
    sorted_crop_list = [crop_list[1:][i] for i in sorted_score_idx][:show_topk]
    # Plotting helper function
    def _plot_subplot(_ax, title='', fw=None):
        _ax.set_title(title, fontweight=fw, fontsize=12)
        _ax.set_xticks([])
        _ax.set_yticks([])
        [x.set_linewidth(2) for x in _ax.spines.values()]
     # Plot query crop
    num_show = len(sorted_crop_list) + 1
    fig, ax = plt.subplots(nrows=1, ncols=num_show, figsize=(plot_width*num_show, plot_width*3))
    ax[0].imshow(crop_list[0].resize((100, 300)))
    _plot_subplot(ax[0], title='Query', fw='bold')
    # Plot gallery crops
    for i, (crop, score) in enumerate(zip(sorted_crop_list, sorted_score_list), 1):
        ax[i].imshow(crop.resize((100, 300)))
        _plot_subplot(ax[i], title='s={:.2f}'.format(score))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        [x.set_linewidth(2) for x in ax[i].spines.values()]
    # Return fig
    return fig

def show_query(query_results_dir, query_image_dir):
    for fname in os.listdir(query_results_dir):
        results_path = os.path.join(query_results_dir, fname)
        with open(results_path, 'r') as fp:
            results_dict = json.load(fp)
        for image_file, _results_dict in results_dict.items():
            image_path = os.path.join(query_image_dir, image_file)
            image = Image.open(image_path)
            show_detects(image, _results_dict, show_det_score=True, figsize=(4, 4))
            query_crop_list = get_crops(image, _results_dict)
    return query_crop_list

def show_gallery(gallery_results_dir, gallery_image_dir, query_crop_list):
    crop_list, score_list = query_crop_list, []
    for fname in os.listdir(gallery_results_dir):
        results_path = os.path.join(gallery_results_dir, fname)
        with open(results_path, 'r') as fp:
            results_dict = json.load(fp)
        for image_file, _results_dict in results_dict.items():
            image_path = os.path.join(gallery_image_dir, image_file)
            image = Image.open(image_path)
            show_detects(image, _results_dict, show_det_score=True, show_sim_score=True)
            crop_list += get_crops(image, _results_dict)
            score_list += _results_dict['person_sim']
            plt.show()
            display(HTML('<hr>'))
    show_reid(crop_list, score_list)
    plt.show()
