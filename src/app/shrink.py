# Global imports
import argparse
import torch


# Main
def main():
    """
    Takes model training checkpoint and removes keys unneeded for inference
    (mainly expensive optimizer state dict).
    """
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_ckpt_path', default='./ckpt/cuhk_final_convnext-base_e30.pkl.old')
    parser.add_argument('--new_ckpt_path', default='./ckpt/cuhk_final_convnext-base_e30.pkl')
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.old_ckpt_path, map_location='cpu')

    # Form new checkpoint dict with subset of keys
    new_ckpt = {
        'model': ckpt['model'],
        'epoch': ckpt['epoch'],
        'config': ckpt['config'],
    }

    # Save new checkpoint dict
    torch.save(new_ckpt, args.new_ckpt_path)
