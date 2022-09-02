# Cert monkey patch for torchvision model download (if needed)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Global imports
import argparse
import os
import shutil
import pickle
from pprint import pprint
import random
from functools import partial
import numpy as np
import yaml
## torch
import torch
import torch.utils.data
import torch.distributed
## ray
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.torch import is_distributed_trainable
from ray.tune.integration.torch import DistributedTrainableCreator
from ray.tune.integration.torch import distributed_checkpoint_dir

# Package imports
## engine
from osr.engine import dist_utils
from osr.engine import transform
from osr.engine.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from osr.engine.train import train_one_epoch
from osr.engine import evaluate
## data
from osr.data import det_utils
## SeqNeXt model
from osr.models.seqnext import SeqNeXt


# Function for reproducible seeding of DataLoader
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Helper to delete keys from torch module state dict
def _del_key(state_dict, key):
    if key in state_dict:
        del state_dict[key]


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


def run(config, checkpoint_dir=None, use_detector=True, detector_only=False):
    # Make training (more) reproducible
    ## We do not control for non-determinism in pytorch, torchvision
    if config['use_random_seed']:
        ## Random seeds
        random.seed(config['random_seed'])
        np.random.seed(config['random_seed'])
        torch.manual_seed(config['random_seed'])
        torch.cuda.manual_seed(config['random_seed'])
        torch.cuda.manual_seed_all(config['random_seed'])
        os.environ["PYTHONHASHSEED"] = str(config['random_seed'])

    # Distributed setup
    if config['debug']:
        print('==> Not using distributed: debug')
        rank = 0
    else:
        print('==> Using distributed: train or test')
        rank = torch.distributed.get_rank()
        dist_utils.setup_for_distributed(rank == 0)

    # Get the device
    device = torch.device(config['device'])

    # Data loading code
    print("Loading data")
    train_loader, num_pid = get_train_loader(config)
    test_loader = get_test_loader(config)

    # Build model
    print("Creating model")
    model, model_without_ddp = get_model(config, num_pid, device=device)

    # Just test for one epoch, no training
    if config['test_only']:
        print('==> Running in test-only mode')
        metric_dict, value_dict = evaluate.evaluate_performance(model, test_loader, device,
            use_amp=config['use_amp'],
            use_gfn=config['use_gfn'],
            gfn_mode=config['gfn_mode']) 

        # Report results
        if config['debug']:
            pprint(metric_dict)
        else:
            ## Store metrics
            tune.report(**metric_dict)

            ## Store values in pickle to conserve space
            trial_dir = os.path.abspath(os.path.join(tune.get_trial_dir(), os.pardir))
            value_path = os.path.join(trial_dir, 'values.pkl')
            with open(value_path, 'wb') as fp:
                pickle.dump(value_dict, fp)

        # Don't do the rest of the main function
        print('==> SUCCESS!!!')
        return

    # Initialize the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    print('Num used params: {}/{}'.format(len(params), len(list(model.parameters()))))
    if config['optimizer'][0] == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=config['optimizer'][1], momentum=0.9, weight_decay=5e-4)
    elif config['optimizer'][0] == 'adam':
        optimizer = torch.optim.Adam(params, lr=config['optimizer'][1])
    else:
        raise Exception

    if config['scheduler'] == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_steps'], gamma=config['lr_gamma'])
    else:
        lr_scheduler = None

    # Train loop
    print("Start training")
    for epoch in range(config['epochs']):
        # Instantiate dict for ray tune report for this epoch
        report_dict = {
            'epoch': epoch,
        }

        # Set scheduler
        if (config['scheduler'] in ('onecycle', 'cosine')) and epoch == 0:
            if config['scheduler'] == 'onecycle':
                print('==> Using OneCycleLR scheduler')
                lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2*config['optimizer'][1],
                    steps_per_epoch=len(train_loader), epochs=config['epochs'], cycle_momentum=False)
            elif config['scheduler'] == 'cosine':
                print('==> Using CosineAnnealing scheduler')
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                    T_max=len(train_loader)*config['epochs'])

        # Train for one epoch
        metric_logger = train_one_epoch(model, optimizer, train_loader,
            device, epoch, config['print_freq'], optimizer_type=config['optimizer'][0],
            warmup=config['use_warmup'],
            use_amp=config['use_amp'], clip_grads=config['clip_grads'],
            lr_scheduler=lr_scheduler if (config['scheduler'] in ('onecycle', 'cosine')) else None)

        # Step the scheduler
        if (config['scheduler'] not in ('onecycle', 'cosine')) and (lr_scheduler is not None):
            lr_scheduler.step()
        report_dict.update(metric_logger.get_dict())

        # Run detection evaluation every test_eval_interval epochs
        if (((epoch + 1) % config['eval_interval']) == 0):

            metric_dict = {}
            try:
                _metric_dict, _ = evaluate.evaluate_performance(model, test_loader, device,
                    use_amp=config['use_amp'],
                    use_gfn=config['use_gfn'], gfn_mode=config['gfn_mode']) 
                metric_dict.update(_metric_dict)
            except RuntimeError:
                raise
            else:
                report_dict.update(metric_dict)

        # Checkpoint
        if ((epoch + 1) % config['ckpt_interval'] == 0) and (epoch > 0):
            if config['debug']:
                ckpt_path = os.path.join('debug_ckpt', '{}_debug_checkpoint_e{}.pkl'.format(config['dataset'], epoch))
                ckpt_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                if lr_scheduler is not None:
                    ckpt_dict['lr_scheduler'] = lr_scheduler.state_dict()
                print('Saving debug checkpoint epoch {} to: {}'.format(epoch, ckpt_path))
                torch.save(ckpt_dict, ckpt_path)
            elif config['log_dir'] and config['ckpt_interval']:
                with distributed_checkpoint_dir(step=epoch) as checkpoint_dir:
                    ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pkl')
                    ckpt_dict = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    if lr_scheduler is not None:
                        ckpt_dict['lr_scheduler'] = lr_scheduler.state_dict()
                    print('Saving checkpoint epoch {} to: {}'.format(epoch, ckpt_path))
                    torch.save(ckpt_dict, ckpt_path)
                    # Manually delete old worker checkpoints because ray can't handle it
                    worker_dir = tune.get_trial_dir()
                    if os.path.exists(worker_dir):
                        for _worker_dir in [os.path.join(worker_dir, d) for d in os.listdir(worker_dir)]:
                            if os.path.basename(_worker_dir).startswith('checkpoint'):
                                old_epoch_str = _worker_dir.split('_')[-1]
                                if old_epoch_str.startswith('tmp'):
                                    shutil.rmtree(_worker_dir)
                                elif int(old_epoch_str) < (epoch - 2):
                                    shutil.rmtree(_worker_dir)
            
        # Report final results for the epoch
        if config['debug']:
            pprint(report_dict)
        else:
            tune.report(**report_dict)


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


# Main function
def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_config', default='./configs/default.yaml')
    parser.add_argument('--trial_config', default='./configs/default.yaml')
    args = parser.parse_args()

    # Load config
    default_config, tuple_key_list = load_config(args.default_config)
    trial_config, _ = load_config(args.trial_config, tuple_key_list=tuple_key_list)
    config = {**default_config, **trial_config}

    # Make log dir
    if config['log_dir']:
        dist_utils.mkdir(config['log_dir'])

    # Debug mode: no need to use ray
    if config['debug']:
        run(config)
    # Use ray
    else:
        # Initialize ray
        ray.init()

        # Make trainable object
        trainable_cls = DistributedTrainableCreator(
            partial(run),
            num_workers=config['num_workers'],
            num_cpus_per_worker=config['num_cpus_per_trial'],
            num_gpus_per_worker=config['num_gpus_per_trial'],
            backend='nccl',
        )

        # Limit the number of rows.
        reporter = CLIReporter(max_progress_rows=10, max_report_frequency=60)

        # Run the trial
        result = tune.run(
            trainable_cls,
            config=config,
            num_samples=config['num_samples'],
            scheduler=None,
            search_alg=None,
            progress_reporter=reporter,
            local_dir=config['log_dir'],
            resume=config['resume'],
            name=config['trial_name'],
            checkpoint_at_end=False,
            max_failures=0,
            keep_checkpoints_num=4,
            checkpoint_score_attr='epoch',
        )


# Run as module
if __name__ == '__main__':
    main()
