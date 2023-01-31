# Global imports
import argparse
import torch
torch._C._jit_set_nvfuser_enabled(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)
# Package imports
from osr.engine import utils as engine_utils


@torch.no_grad()
def main():
    """
    Test docstring...
    """
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_config', default='./configs/default.yaml')
    parser.add_argument('--trial_config', default='./configs/cuhk_test_final.yaml')
    parser.add_argument('--torchscript_path', default='./torchscript/cuhk_final_convnext-base_e30.torchscript.pt')
    args = parser.parse_args()

    # Load config
    default_config, tuple_key_list = engine_utils.load_config(args.default_config)
    trial_config, _ = engine_utils.load_config(args.trial_config, tuple_key_list=tuple_key_list)
    config = {**default_config, **trial_config}
    config['debug'] = True
    device = torch.device(config['device'])

    # Load model
    _, model = engine_utils.get_model(config, num_pid=0, device=device)
    model.eval()

    # Generate fake input
    w, h = 1500, 900
    images = [torch.randn(3, h, w).to(device)]
    targets = ({'boxes': torch.FloatTensor([[0.0, 0.0, w, h]]).to(device)},)

    # Put fake input through model
    print('==> Building torchscript...')
    model.forward = model.inference
    script = torch.jit.script(model, {'images': images, 'targets': targets, 'inference_mode': 'both'})

    # Save torchscript
    print('==> Saving torchscript...')
    torch.jit.save(script, args.torchscript_path)
