# Global imports
import os
import yaml
import random
import numpy as np
## torch
import torch
import torch.utils.data
## ray
from ray import tune
from ray.tune.integration.torch import is_distributed_trainable

# Package imports
## SeqNeXt model
from osr.models.seqnext import SeqNeXt
## engine
from osr.engine import transform
from osr.engine.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from osr.engine import dist_utils
## data
from osr.data import det_utils


# Helper function to move data to GPU
def to_device(images, targets, device):
    images = [image.to(device) for image in images]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    return images, targets


# Helper to delete keys from torch module state dict
def _del_key(state_dict, key):
    if key in state_dict:
        del state_dict[key]


# YAML config loader function
def load_config(path, tuple_key_list=None):
    # Load config dict from YAML
    with open(path, 'r') as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)
            raise

    # Params that have tuple type
    if tuple_key_list is None:
        tuple_key_list = config['tuple_key_list']
        del config['tuple_key_list']
    elif 'tuple_key_list' in config:
        del config['tuple_key_list']
    
    # Convert lists to tune.grid_search
    proc_config = {}
    for key, val in config.items():
        if key in tuple_key_list:
            if type(val) == list:
                proc_config[key] = tune.grid_search([eval(_val) for _val in val])
            else:
                proc_config[key] = eval(val)
        else:
            if type(val) == list:
                proc_config[key] = tune.grid_search(val)
            else:
                proc_config[key] = val

    # Return processed config dict
    return proc_config, tuple_key_list


# Load the model
def get_model(config, num_pid, device='cpu'):
    # Build SeqNeXt model
    model = SeqNeXt(config, oim_lut_size=num_pid, device=device)

    # Save model repr for reference
    ## Useful diagnostic when tinkering with model layers
    if not config['debug']:
        trial_dir = os.path.abspath(os.path.join(tune.get_trial_dir(), os.pardir))
        model_repr_path = os.path.join(trial_dir, 'model_repr.txt')
        with open(model_repr_path, 'w') as fp:
            fp.write(model.__repr__())

    # Put model on GPU
    print('Cuda available:', torch.cuda.is_available())
    model.to(device)

    # Make model distributed
    if is_distributed_trainable():
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Load checkpoint if it is available
    if config['ckpt_path']:
        print('==> Restoring checkpoint from path:', config['ckpt_path'])
        checkpoint = torch.load(config['ckpt_path'], map_location='cpu')
        state_dict = checkpoint['model']
        # Delete keys with potentially conflicting params
        _del_key(state_dict, 'roi_heads.reid_loss.lut')
        _del_key(state_dict, 'roi_heads.reid_loss.cq')
        _del_key(state_dict, 'gfn.reid_loss.lut')
        _del_key(state_dict, 'gfn.reid_loss.cq')
        _del_key(state_dict, 'roi_heads.gfn.reid_loss.lut')
        _del_key(state_dict, 'roi_heads.gfn.reid_loss.cq')
        # Load state dict into the model
        model_without_ddp.load_state_dict(state_dict, strict=False)

    #
    return model, model_without_ddp


# Function for reproducible seeding of DataLoader
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_test_loader(config):
    # Use ImageNet stats to standardize the data
    stat_dict = {
        'mean': config['image_mean'],
        'std': config['image_std'],
    }

    # Set transform
    ## IFN transform
    if config['aug_mode'] == 'wrs':
        test_transform = transform.get_transform_wrs(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rrc':
        test_transform = transform.get_transform_rrc(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rrc2':
        test_transform = transform.get_transform_rrc2(train=False, stat_dict=stat_dict)

    # Get dataset
    test_dataset, _ = det_utils.get_coco(
        config['dataset_dir'], dataset_name=config['dataset'], image_set=config['test_set'], transforms=test_transform)

    # Test sampler
    if config['debug']:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    else:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    # Get test dataset objects
    retrieval_dir = os.path.join(config['dataset_dir'], 'retrieval')
    test_sampler = det_utils.TestSampler(config['test_set'], test_dataset, retrieval_dir,
        config['retrieval_name_list'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        sampler=test_sampler, num_workers=config['workers'],
        collate_fn=dist_utils.collate_fn)

    #
    return test_loader


def get_train_loader(config):
    # Use ImageNet stats to standardize the data
    stat_dict = {
        'mean': config['image_mean'],
        'std': config['image_std'],
    }

    # Set transform
    ## IFN transform
    if config['aug_mode'] == 'wrs':
        train_transform = transform.get_transform_wrs(train=True, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rrc':
        train_transform = transform.get_transform_rrc(train=True, stat_dict=stat_dict,
            rfc_prob=config['aug_rfc_prob'], rsc_prob=config['aug_rsc_prob'],
            rbsc_er=config['aug_rbsc_er'], crop_res=config['aug_crop_res'])
    elif config['aug_mode'] == 'rrc2':
        train_transform = transform.get_transform_rrc2(train=True, stat_dict=stat_dict,
            rfc_prob=config['aug_rfc_prob'], rsc_prob=config['aug_rsc_prob'],
            rbsc_er=config['aug_rbsc_er'], crop_res=config['aug_crop_res'])

    # Get dataset
    ## Load train dataset even in test only mode, because we need to get the pid_lookup_dict to load the correct OIM
    train_dataset, pid_lookup_dict = det_utils.get_coco(
        config['dataset_dir'], dataset_name=config['dataset'], image_set=config['train_set'], transforms=train_transform)
    
    # Train sampler
    print("Creating data loaders")
    train_sampler = torch.utils.data.SequentialSampler(train_dataset)

    # Get train data loader
    if config['debug']:
        print('WARNING: sequential sampler for debugging')
        train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    else:
        print('Using random sampler...')
        ## Control randomness
        if config['use_random_seed']:
            g_rs = torch.Generator()
            g_rs.manual_seed(config['random_seed'])
        else:
            g_rs = None
        ## Build train sampler
        train_sampler = torch.utils.data.RandomSampler(train_dataset, generator=g_rs)

    # Determine aspect ratio grouping based on dataset
    if config['dataset'] == 'cuhk':
        if config['aspect_ratio_grouping']:
            print('Using aspect ratio batch sampler...')
            ## Group into two bins: wide and tall
            ### Saves considerable memory with WindowResize transform, allowing for larger batch size
            aspect_ratio_group_factor = 0
        else:
            aspect_ratio_group_factor = -1
    elif config['dataset'] == 'prw':
        ## AR grouping does not benefit PRW: 5/6 cameras have same image size
        aspect_ratio_group_factor = -1
    else:
        raise NotImplementedError

    # Setup aspect ratio batch sampler
    if aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, config['batch_size'])
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, config['batch_size'], drop_last=True)
    # Set up train loader
    ## Control randomness
    if config['use_random_seed']:
        g_dl = torch.Generator()
        g_dl.manual_seed(config['random_seed'])
        worker_init_fn = _seed_worker
    else:
        g_dl = None
        worker_init_fn = None
    ## Build train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler,
        num_workers=config['workers'],
        collate_fn=dist_utils.collate_fn,
        generator=g_dl, worker_init_fn=worker_init_fn)

    #
    return train_loader, pid_lookup_dict['num_pid']


