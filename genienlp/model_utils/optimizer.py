import math
from functools import partial

import numpy as np
import torch
from pytorch_lightning_spells.lr_schedulers import LinearLR, MultiStageScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    Adafactor,
    AdamW,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


class MultiStageSchedulerV2(MultiStageScheduler):
    def __init__(self, schedulers, start_at_epochs, last_epoch=-1):
        super().__init__(schedulers, start_at_epochs, last_epoch)

    def get_last_lr(self, epoch=None):
        if epoch is None:
            self.last_epoch = self.last_epoch + 1
        else:
            self.last_epoch = epoch - 1
        for scheduler, starting_epoch in zip(self.schedulers, self.start_at_epochs):
            if self.last_epoch + 1 >= starting_epoch:
                scheduler.last_epoch = self.last_epoch - starting_epoch
                return scheduler.get_last_lr()


def get_transformer_learning_rate(i, *, dimension, warmup):
    i += 1
    return 1.0 / math.sqrt(dimension) * min(1 / math.sqrt(i), i / (warmup * math.sqrt(warmup)))


def get_sgd_learning_rate(i, *, warmup):
    i += 1
    return min(math.sqrt(warmup) / math.sqrt(i), i / warmup)


def init_opt(args, model, logger):
    num_training_steps = sum(args.train_iterations) // args.gradient_accumulation_steps

    if args.optimizer == 'adam':
        # Adam with transformer schedule has a different set of default hyperparameters:
        if args.lr_schedule == 'transformer':
            opt = torch.optim.Adam(
                model.params, lr=args.lr_multiply, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay
            )
        else:
            opt = torch.optim.Adam(
                model.params, lr=args.lr_multiply, betas=(args.beta0, 0.999), weight_decay=args.weight_decay
            )
    elif args.optimizer == 'adamw':
        opt = AdamW(model.params, lr=args.lr_multiply, weight_decay=args.weight_decay)
    elif args.optimizer == 'adafactor':
        opt = Adafactor(
            model.params, lr=args.lr_multiply, weight_decay=args.weight_decay, relative_step=False, scale_parameter=False
        )
    elif args.optimizer == 'radam':
        import radam

        if args.warmup > 1:
            logger.warning('With RAdam optimizer, warmup is never applied')
        opt = radam.RAdam(model.params, lr=args.lr_multiply, betas=(args.beta0, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        opt = torch.optim.SGD(model.params, lr=args.lr_multiply, weight_decay=args.weight_decay)
    else:
        raise ValueError('Invalid optimizer.')

    if args.lr_schedule == 'transformer':
        lr_lambda = partial(get_transformer_learning_rate, dimension=args.dimension, warmup=args.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    elif args.lr_schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=args.warmup)
    elif args.lr_schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            opt,
            num_training_steps=num_training_steps,
            num_warmup_steps=args.warmup,
        )
    elif args.lr_schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_training_steps=num_training_steps,
            num_warmup_steps=args.warmup,
            num_cycles=0.5,
        )
    elif args.lr_schedule == 'multi_linear_cosine':
        lr_durations = [int(num_training_steps * 0.1), int(np.ceil(num_training_steps * 0.9)) + 1]
        start_at_epochs = [0] + list(np.cumsum(lr_durations))[:-1]
        linear_scheduler = LinearLR(opt, 0.0001, lr_durations[0])
        cosine_scheduler = CosineAnnealingLR(opt, lr_durations[1])
        scheduler = MultiStageSchedulerV2([linear_scheduler, cosine_scheduler], start_at_epochs)
    elif args.lr_schedule == 'sgd':
        lr_lambda = partial(get_sgd_learning_rate, warmup=args.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        raise ValueError('Invalid learning rate scheduler.')

    return opt, scheduler
