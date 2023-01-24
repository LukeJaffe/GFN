# Global imports
import os
import math
import numpy as np
import torch
import collections

# Local imports
from osr.engine import dist_utils


# Function to train for one epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, optimizer_type=None, warmup=None, use_amp=True, lr_scheduler=None, clip_grads=False, max_norm=10.0):
    model.train()
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # Make sure we are not using warmup and scheduler
    assert not (warmup and (lr_scheduler is not None))

    # Warmup init
    if warmup and (epoch == 0):
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # Set up scaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Train loop
    for train_iter, (_images, _targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        try:
            # Move images and targets to GPU
            images = list(image.to(device) for image in _images)
            targets = [{k:(v.to(device) if type(v) != list else v) for k, v in t.items()} for t in _targets]

            # Standard SGD forward pass
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp): 
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
            if use_amp:
                scaler.scale(losses).backward()
            else:
                losses.backward()

            # Update weights using stored gradients
            if use_amp:
                ## Clip grad norm
                if clip_grads:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                # Update scaler
                scaler.update()
            else:
                ## Clip grad norm
                if clip_grads:
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
                optimizer.step()

            # Get losses
            with torch.no_grad():
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                # Get loss value
                loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Some loss not caught is: {}, continue training".format(loss_value))
                print(loss_dict_reduced)
                raise RuntimeError

            # Step the LR scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Update logger
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # Empty GPU cache
            torch.cuda.empty_cache()

        # Catch loss \in {nan, inf}, and RuntimeError: CUDA out of memory.
        except RuntimeError as e:
            # Empty GPU cache
            torch.cuda.empty_cache()

            # Raise the exception
            raise

    # Return metrics
    return metric_logger
