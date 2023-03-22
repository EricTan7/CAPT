import torch
import torch.nn as nn
import torch.distributed as dist
# from apex import amp
from torch.cuda.amp import GradScaler, autocast

from tools.meter import AverageMeter, MetricMeter
from tools.dist_utils import reduce_value
from tools.metrics import compute_accuracy
from solver import Classification

import time
import datetime
import logging
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix

device = 'cuda'


def train(cfg, model, data, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim

    if device:
        model.to(local_rank)
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    num_batches = len(data.train_loader)
    tot_epoch = cfg.OPTIM.MAX_EPOCH

    # 1. loss
    criterion = nn.CrossEntropyLoss()

    # 2. meter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    for epoch in range(1, tot_epoch+1):
        loss_meter.reset()
        acc_meter.reset()
        batch_time.reset()

        # 3. train
        for batch_idx, batch in enumerate(data.train_loader):
            end = time.time()
            image, label = parse_batch(batch)
            # forward
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # amp training
            # if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            #     with autocast(enabled=True):
            #         output = model(image)
            #         loss = criterion(output, label)
            #
            #     scaler.scale(loss).backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     loss = reduce_value(loss, average=True)
            # else:
            #     output = model(image)
            #     loss = criterion(output, label)
            #     optimizer.step()

            if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                loss = reduce_value(loss, average=True)

            acc = compute_accuracy(output, label, topk=(1,))[0].item()

            loss_meter.update(loss.item(), image.shape[0])
            acc_meter.update(acc, 1)

            # compute time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            meet_freq = (batch_idx + 1) % cfg.TRAIN.PRINT_FREQ == 0
            if meet_freq:
                if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                    pass
                else:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (tot_epoch - epoch) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"Epoch [{epoch}/{tot_epoch}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})"]
                    info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                    if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                        info += [f"lr {model.module.get_current_lr():.4e}"]
                    else:
                        info += [f"lr {model.get_current_lr():.4e}"]
                    info += [f"eta {eta}"]
                    logger.info(" ".join(info))

        # 1.update lr
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            model.module.update_lr()
        else:
            model.update_lr()

        # 2.meet epoch: save checkpoint
        sdir = cfg.OUTPUT_DIR
        if epoch % cfg.TRAIN.SAVE_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.save_model(epoch, sdir, is_best=False)
                else:
                    model.save_model(epoch, sdir, is_best=False)

        # 3.meet epoch: do test
        if epoch % cfg.TRAIN.TEST_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    # model.module.set_model_mode("test")
                    model.eval()
                    test(cfg, model, data)
                    model.train()
                    # model.module.set_model_mode("train")
                else:
                    # model.set_model_mode("test")
                    model.eval()
                    test(cfg, model, data)
                    model.train()
                    # model.set_model_mode("train")

        # if (self.batch_idx + 1) == self.num_batches:
        #     self.log_ctx.append(self.model.prompt_learner.ctx[0, :16].clone().detach())
        #     self.log_meta.append(self.model.prompt_learner.meta_net.linear1.weight.data[0, :16].clone().detach())
        #     self.log_head.append(self.model.cls_head.fc.weight.data[0, :16].clone().detach())
        #     with open('ctx_baseline_addhead.txt', 'w') as f:
        #         for item in self.log_ctx:
        #             f.write(str(item)+'\n')
        #
        #     with open('meta_baseline_addhead.txt', 'w') as f:
        #         for item in self.log_meta:
        #             f.write(str(item)+'\n')
        #
        #     with open('head_baseline_addhead.txt', 'w') as f:
        #         for item in self.log_head:
        #             f.write(str(item)+'\n')


# def test(model, test_loader):
#     acc_meter = AverageMeter()
#     for iter, batch in tqdm(enumerate(test_loader)):
#         with torch.no_grad():
#             image, label = parse_batch(batch)
#             output = model(image)
#             acc = compute_accuracy(output, label, topk=(1,))[0].item()
#             acc_meter.update(acc, 1)
#
#     return acc_meter.avg

def test(cfg, model, data):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    evaluator = Classification(cfg, logger)

    logger.info(f"Evaluate on the *test* set")

    for batch_idx, batch in enumerate(tqdm(data.test_loader)):
        with torch.no_grad():
            image, label = parse_batch(batch)
            output = model(image)
            evaluator.process(output, label)

    return evaluator.evaluate()     # results["accuracy"]


def parse_batch(batch):
    input = batch["img"]
    label = batch["label"]
    input = input.to(device)
    label = label.to(device)
    return input, label


def train_wandb(cfg, model, data, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim

    if device:
        model.to(local_rank)
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    num_batches = len(data.train_loader)
    tot_epoch = cfg.OPTIM.MAX_EPOCH

    # 1. loss
    criterion = nn.CrossEntropyLoss()

    # 2. meter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    for epoch in range(1, tot_epoch + 1):
        loss_meter.reset()
        acc_meter.reset()
        batch_time.reset()

        # 3. train
        for batch_idx, batch in enumerate(data.train_loader):
            end = time.time()
            image, label = parse_batch(batch)
            # forward
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                loss = reduce_value(loss, average=True)

            acc = compute_accuracy(output, label, topk=(1,))[0].item()

            loss_meter.update(loss.item(), image.shape[0])
            acc_meter.update(acc, 1)

            # compute time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            meet_freq = (batch_idx + 1) % cfg.TRAIN.PRINT_FREQ == 0
            if meet_freq:
                if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                    pass
                else:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (tot_epoch - epoch) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"Epoch [{epoch}/{tot_epoch}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})"]
                    info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                    if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                        info += [f"lr {model.module.get_current_lr():.4e}"]
                    else:
                        info += [f"lr {model.get_current_lr():.4e}"]
                    info += [f"eta {eta}"]
                    logger.info(" ".join(info))

        wandb.log({'train loss': loss_meter.avg,
                   'train acc': acc_meter.avg,
                   'epoch': epoch
                   })

        # 1.update lr
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            model.module.update_lr()
        else:
            model.update_lr()

        # 2.meet epoch: save checkpoint
        sdir = cfg.OUTPUT_DIR
        if epoch % cfg.TRAIN.SAVE_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.save_model(epoch, sdir, is_best=False)
                else:
                    model.save_model(epoch, sdir, is_best=False)

        # 3.meet epoch: do test
        if epoch % cfg.TRAIN.TEST_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.set_model_mode("test")
                    results = test(cfg, model, data)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    results = test(cfg, model, data)
                    model.set_model_mode("train")
                wandb.log({'test acc': results["accuracy"]})


def train_sweep(cfg, model, data, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim

    if device:
        model.to(local_rank)
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    num_batches = len(data.train_loader)
    tot_epoch = cfg.OPTIM.MAX_EPOCH

    # 1. loss
    criterion = nn.CrossEntropyLoss()

    # 2. meter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    for epoch in range(1, tot_epoch+1):
        loss_meter.reset()
        acc_meter.reset()
        batch_time.reset()

        # 3. train
        for batch_idx, batch in enumerate(data.train_loader):
            end = time.time()
            image, label = parse_batch(batch)
            # forward
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                loss = reduce_value(loss, average=True)

            acc = compute_accuracy(output, label, topk=(1,))[0].item()

            loss_meter.update(loss.item(), image.shape[0])
            acc_meter.update(acc, 1)

            # compute time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            meet_freq = (batch_idx + 1) % cfg.TRAIN.PRINT_FREQ == 0
            if meet_freq:
                if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                    pass
                else:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (tot_epoch - epoch) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"Epoch [{epoch}/{tot_epoch}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})"]
                    info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                    if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                        info += [f"lr {model.module.get_current_lr():.4e}"]
                    else:
                        info += [f"lr {model.get_current_lr():.4e}"]
                    info += [f"eta {eta}"]
                    logger.info(" ".join(info))

        wandb.log({'train loss': loss_meter.avg,
                   'train acc': acc_meter.avg
                   })

        # 1.update lr
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            model.module.update_lr()
        else:
            model.update_lr()

        # 2.meet epoch: save checkpoint
        sdir = cfg.OUTPUT_DIR
        if epoch % cfg.TRAIN.SAVE_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.save_model(epoch, sdir, is_best=False)
                else:
                    model.save_model(epoch, sdir, is_best=False)

        # 3.meet epoch: do test
        if epoch % cfg.TRAIN.TEST_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.eval()
                    results = test(cfg, model, data)
                    model.train()
                else:
                    model.eval()
                    results = test(cfg, model, data)
                    model.train()
                wandb.log({'test acc': results["accuracy"]})