import argparse
import os
import time

import wandb
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from models import UNet
from loss_functions import dispatch_loss_function, compute_loss
from training import warmup, dispatch_lr_scheduler, get_lr, dispatch_optimizer
from dataset import get_train_dataloader, get_test_dataloader
from utils import parse_args


if __name__ == '__main__':
    args = parse_args()
    use_cuda = not args.use_cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    bs = args.train_batch_size

    train_dataloader = get_train_dataloader(args.data_dir, args.train_batch_size)
    test_dataloader = get_test_dataloader(args.data_dir, args.test_batch_size)

    model = UNet().to(device)

    wandb.init(project="semantic-segmentation", config=args) # TODO
    wandb.watch(model, log='all')
    config = wandb.config

    loss_function = dispatch_loss_function(args.loss_function)
    optimizer = dispatch_optimizer(model, args)
    lr_scheduler = dispatch_lr_scheduler(optimizer, args)

    iteration = 0
    # training_accuracy = compute_accuracy(model, train_dataloader, device)
    # test_accuracy = compute_accuracy(model, test_dataloader, device)
    # wandb.log({'training accuracy': training_accuracy}, step=iteration*bs)
    # wandb.log({'test_accuracy': test_accuracy}, step=iteration*bs)

    for epoch in range(args.epochs):
        for image, true_mask in train_dataloader:
            start_time = time.time()
            if iteration < args.warmup:
                warmup(iteration, optimizer, args.learning_rate, args.warmup)
            image, true_mask = image.to(device), true_mask.to(device)
            predicted_mask = model(image)
            loss = loss_function(true_mask, predicted_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'training loss': loss}, step=iteration*bs)
            wandb.log({'learning rate': get_lr(optimizer)}, step=iteration*bs)

            wandb.log({'iteration': iteration}, step=iteration * bs)
            wandb.log({'iteration time': time.time() - start_time}, step=iteration*bs)
            if iteration % 10 == 0:
                test_loss = compute_loss(model, test_dataloader, loss_function, device)
                wandb.log({'test loss': loss}, step=iteration*bs)
            iteration += 1

        lr_scheduler.step()
        # training_accuracy = compute_accuracy(model, train_dataloader, device)
        # test_accuracy = compute_accuracy(model, test_dataloader, device)
        # wandb.log({'training accuracy': training_accuracy}, step=iteration*bs)
        # wandb.log({'test_accuracy': test_accuracy}, step=iteration * bs)
