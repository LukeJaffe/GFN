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
import numpy as np
from functools import partial
## torch
import torch
import torch.distributed
## ray
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.torch import DistributedTrainableCreator
from ray.tune.integration.torch import distributed_checkpoint_dir

# Package imports
## engine
from osr.engine import dist_utils
from osr.engine.train import train_one_epoch
from osr.engine import evaluate
from osr.engine import utils as engine_utils


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
    train_loader, num_pid = engine_utils.get_train_loader(config)
    test_loader = engine_utils.get_test_loader(config)

    # Build model
    print("Creating model")
    model, model_without_ddp = engine_utils.get_model(config, num_pid, device=device)

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


# Main function
def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_config', default='./configs/default.yaml')
    parser.add_argument('--trial_config', default='./configs/default.yaml')
    args = parser.parse_args()

    # Load config
    default_config, tuple_key_list = engine_utils.load_config(args.default_config)
    trial_config, _ = engine_utils.load_config(args.trial_config, tuple_key_list=tuple_key_list)
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
