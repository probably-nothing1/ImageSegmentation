import argparse
import os
import time

import wandb
import torch
import torch.nn.functional as F

from models import UNet
from loss_functions import dispatch_loss_function, compute_loss, dispatch_second_loss_function
from training import warmup, dispatch_lr_scheduler, get_lr, dispatch_optimizer
from dataset import get_train_dataloader, get_test_dataloader
from metrics import compute_metrics
from utils import parse_args, save_model

best_acc = 0
save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

def compute_and_log_metrics(model, train_dataloader, test_dataloader, augmentations, device, step, dataset_version):
    training_accuracy, training_IoU = compute_metrics(model, train_dataloader, augmentations, device, dataset_version=dataset_version)
    test_accuracy, test_IoU = compute_metrics(model, test_dataloader, augmentations, device, dataset_version=dataset_version)
    wandb.log({'training IoU': training_IoU, 'training accuracy': training_accuracy}, step=iteration*bs)
    wandb.log({'test IoU': test_IoU, 'test accuracy': test_accuracy}, step=iteration * bs)
    return test_accuracy, test_IoU

if __name__ == '__main__':
    args = parse_args()
    print(args)

    use_cuda = not args.use_cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    bs = args.train_batch_size
    dataset_version = args.dataset_version
    eval_batch = 2 if dataset_version == 'flip' else (1 if dataset_version == 'single' else 6)

    train_dataloader = get_train_dataloader(args.data_dir, args.train_batch_size, dataset_version, shuffle=True, use_transforms=args.augmentations)
    test_dataloader = get_test_dataloader(args.data_dir, args.test_batch_size, dataset_version, shuffle=True, use_transforms=args.augmentations)

    metrics_train_dataloader = get_train_dataloader(args.data_dir, eval_batch, dataset_version, shuffle=False, use_transforms=False)
    metrics_test_dataloader = get_test_dataloader(args.data_dir, eval_batch, dataset_version, shuffle=False, use_transforms=False)

    model = UNet().to(device)

    wandb.init(project=args.project_name, name=args.run_name, config=args)
    wandb.watch(model, log='all')
    config = wandb.config

    loss_function = dispatch_loss_function(args)
    second_loss_function = dispatch_second_loss_function(args)
    optimizer = dispatch_optimizer(model, args)
    lr_scheduler = dispatch_lr_scheduler(optimizer, args)

    iteration = 0
    compute_and_log_metrics(model, metrics_train_dataloader, metrics_test_dataloader, args.augmentations, device, iteration*bs, dataset_version)

    for epoch in range(args.epochs):
        print(f'epoch {epoch}')
        for image, true_mask in train_dataloader:
            start_time = time.time()
            if iteration < args.warmup:
                warmup(iteration, optimizer, args.learning_rate, args.warmup)
            image, true_mask = image.to(device), true_mask.to(device)
            pixel_probabilities = model(image)
            loss = loss_function(pixel_probabilities, true_mask)
            if second_loss_function is not None:
                second_loss = second_loss_function(pixel_probabilities, true_mask)
                loss += second_loss * 3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'training loss': loss}, step=iteration*bs)
            wandb.log({'learning rate': get_lr(optimizer)}, step=iteration*bs)

            wandb.log({'iteration': iteration}, step=iteration * bs)
            wandb.log({'iteration time': (time.time() - start_time) / bs}, step=iteration*bs)
            if iteration % 10 == 0:
                test_loss = compute_loss(model, test_dataloader, loss_function, device)
                wandb.log({'test loss': loss}, step=iteration*bs)
            iteration += 1

        lr_scheduler.step()
        test_acc, test_iou = compute_and_log_metrics(model, metrics_train_dataloader, metrics_test_dataloader, args.augmentations, device, iteration*bs, dataset_version)
        if test_acc > best_acc:
            save_model(model, f'{save_dir}/{args.run_name}_{test_acc}.pt')
