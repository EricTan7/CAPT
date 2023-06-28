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
import os
import json
import logging
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix
from copy import deepcopy

device = 'cuda'


def train(cfg, model, data, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim
    optimizer_fc = None
    try:
        optimizer_fc = model.optim_fc
    except:
        pass

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
    prompts_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    for epoch in range(1, tot_epoch+1):
        loss_meter.reset()
        prompts_loss_meter.reset()
        acc_meter.reset()
        batch_time.reset()
        for _ in range(cfg.INPUT.NUM_VIEWS):
            # 3. train
            for batch_idx, batch in enumerate(data.train_loader):
                end = time.time()
                image, label = parse_batch(batch)
                # forward
                # output = model(image, label)
                # loss = criterion(output, label)
                output, loss_prompts = model(image, label)
                loss = criterion(output, label) + loss_prompts
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if optimizer_fc is not None:
                    optimizer_fc.step()
                    optimizer_fc.zero_grad()

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
                prompts_loss_meter.update(loss_prompts.item(), image.shape[0])
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
                        info += [f"prompts_loss {prompts_loss_meter.val:.3f} ({prompts_loss_meter.avg:.3f})"]
                        info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                            info += [f"lr {model.module.get_current_lr():.4e}"]
                        else:
                            info += [f"lr {model.get_current_lr():.4e}"]
                        info += [f"eta {eta}"]
                        logger.info(" ".join(info))
            # log_ppt_query, log_ppt_bonder_cattn_q, log_ppt_bonder_cattn_proj, log_ppt_bonder_sattn_qkv, log_ppt_bonder_sattn_proj, log_clshead = [], [], [], [], [], []
            #
            # log_ppt_query.append(model.model.prompt_learner.query.data.clone().detach())
            # log_ppt_bonder_cattn_q.append(model.model.prompt_learner.bonder.cross_attn.q.weight.data.clone().detach())
            # log_ppt_bonder_cattn_proj.append(model.model.prompt_learner.bonder.cross_attn.proj.weight.data.clone().detach())
            # log_ppt_bonder_sattn_qkv.append(model.model.prompt_learner.bonder.self_attn_layer.qkv.weight.data.clone().detach())
            # log_ppt_bonder_sattn_proj.append(model.model.prompt_learner.bonder.self_attn_layer.proj.weight.data.clone().detach())
            # log_clshead.append(model.model.cls_head.fc.weight.data.clone().detach())
            # with open('log_ppt_query.txt', 'a') as f:
            #     for item in log_ppt_query:
            #         f.write(str(item) + '\n')
            # with open('log_ppt_bonder_cattn_q.txt', 'a') as f:
            #     for item in log_ppt_bonder_cattn_q:
            #         f.write(str(item) + '\n')
            # with open('log_ppt_bonder_cattn_proj.txt', 'a') as f:
            #     for item in log_ppt_bonder_cattn_proj:
            #         f.write(str(item) + '\n')
            # with open('log_ppt_bonder_sattn_qkv.txt', 'a') as f:
            #     for item in log_ppt_bonder_sattn_qkv:
            #         f.write(str(item) + '\n')
            # with open('log_ppt_bonder_sattn_proj.txt', 'a') as f:
            #     for item in log_ppt_bonder_sattn_proj:
            #         f.write(str(item) + '\n')
            # with open('log_clshead.txt', 'a') as f:
            #     for item in log_clshead:
            #         f.write(str(item) + '\n')

            # model.model.prompt_learner.query.data
            # model.model.prompt_learner.bonder.cross_attn.q.weight.data
            # model.model.prompt_learner.bonder.cross_attn.proj.weight.data
            # model.model.prompt_learner.bonder.self_attn_layer.qkv.weight.data
            # model.model.prompt_learner.bonder.self_attn_layer.proj.weight.data
            # model.model.cls_head.fc.weight.data


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
                    test(cfg, model, data)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    test(cfg, model, data)
                    model.set_model_mode("train")

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


def train_wandb(cfg, model, data, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim
    optimizer_fc = None
    try:
        optimizer_fc = model.optim_fc
    except:
        pass

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
    prompts_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    for epoch in range(1, tot_epoch + 1):
        loss_meter.reset()
        prompts_loss_meter.reset()
        acc_meter.reset()
        batch_time.reset()

        for _ in range(cfg.INPUT.NUM_VIEWS):
            # 3. train
            for batch_idx, batch in enumerate(data.train_loader):
                end = time.time()
                image, label = parse_batch(batch)
                # forward
                # output = model(image, label)
                # loss = criterion(output, label)
                output, loss_prompts = model(image, label)
                loss = criterion(output, label) + loss_prompts
                # loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if optimizer_fc is not None:
                    optimizer_fc.step()
                    optimizer_fc.zero_grad()

                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    loss = reduce_value(loss, average=True)

                acc = compute_accuracy(output, label, topk=(1,))[0].item()

                loss_meter.update(loss.item(), image.shape[0])
                prompts_loss_meter.update(loss_prompts.item(),image.shape[0])
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
                        info += [f"prompts_loss {prompts_loss_meter.val:.3f} ({prompts_loss_meter.avg:.3f})"]
                        info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                            info += [f"lr {model.module.get_current_lr():.4e}"]
                        else:
                            info += [f"lr {model.get_current_lr():.4e}"]
                        info += [f"eta {eta}"]
                        logger.info(" ".join(info))

        wandb.log({'train loss': loss_meter.avg,
                   'train acc': acc_meter.avg,
                   'train prompts loss': prompts_loss_meter.avg,
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
                    results, test_loss = test(cfg, model, data)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    results, test_loss = test(cfg, model, data)
                    model.set_model_mode("train")
                wandb.log({'test acc': results["accuracy"],
                           'test loss': test_loss})


def train_wandb_iter(cfg, model, data, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim
    scheduler = model.sched
    optimizer_fc = None
    try:
        optimizer_fc = model.optim_fc
    except:
        pass

    if device:
        model.to(local_rank)
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    tot_iter = cfg.OPTIM.MAX_ITER

    # 1. loss
    criterion = nn.CrossEntropyLoss()

    # 2. meter()
    loss_meter = AverageMeter()
    prompts_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    image_loader_iter = iter(data.train_loader)

    for iters in range(1, tot_iter + 1):
        start = time.time()
        # update lr
        scheduler.step()
        try:
            image, label = parse_batch(next(image_loader_iter))
        except StopIteration:
            image_loader_iter = iter(data.train_loader)
            image, label = parse_batch(next(image_loader_iter))

        output, loss_prompts = model(image, label)
        loss = criterion(output, label) + loss_prompts
        # loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if optimizer_fc is not None:
            optimizer_fc.step()
            optimizer_fc.zero_grad()

        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            loss = reduce_value(loss, average=True)

        acc = compute_accuracy(output, label, topk=(1,))[0].item()

        loss_meter.update(loss.item(), image.shape[0])
        prompts_loss_meter.update(loss_prompts.item(), image.shape[0])
        acc_meter.update(acc, 1)

        batch_time.update(time.time() - start)

        # log lr
        wandb.log({'lr': model.get_current_lr()})

        meet_freq = iters % cfg.TRAIN.PRINT_FREQ == 0
        if meet_freq:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                nb_remain = tot_iter - iters
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"Iter [{iters}/{tot_iter}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})"]
                info += [f"prompts_loss {prompts_loss_meter.val:.3f} ({prompts_loss_meter.avg:.3f})"]
                info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    info += [f"lr {model.module.get_current_lr():.4e}"]
                else:
                    info += [f"lr {model.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))

            wandb.log({'train loss': loss_meter.val,
                       'train acc': acc_meter.val,
                       'train prompts loss': prompts_loss_meter.val,
                       'iter': iters
                       })

        # 2.meet epoch: save checkpoint
        sdir = cfg.OUTPUT_DIR
        if iters % cfg.TRAIN.SAVE_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.save_model(iters, sdir, is_best=False)
                else:
                    model.save_model(iters, sdir, is_best=False)

        # 3.meet epoch: do test
        if iters % cfg.TRAIN.TEST_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.set_model_mode("test")
                    results, test_loss = test(cfg, model, data)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    results, test_loss = test(cfg, model, data)
                    model.set_model_mode("train")
                wandb.log({'test acc': results["accuracy"],
                           'test loss': test_loss})


def train_wandb_iter_wiseft(cfg, model, data, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim
    scheduler = model.sched
    optimizer_fc = None
    try:
        optimizer_fc = model.optim_fc
    except:
        pass

    if device:
        model.to(local_rank)
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    tot_iter = cfg.OPTIM.MAX_ITER

    # 1. loss
    criterion = nn.CrossEntropyLoss()

    # 2. meter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    prompts_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    image_loader_iter = iter(data.train_loader)

    for iters in range(1, tot_iter + 1):
        start = time.time()
        # update lr
        scheduler.step()
        try:
            image, label = parse_batch(next(image_loader_iter))
        except StopIteration:
            image_loader_iter = iter(data.train_loader)
            image, label = parse_batch(next(image_loader_iter))

        output, loss_prompts = model(image, label)
        loss = criterion(output, label) + loss_prompts
        loss_cls = loss - loss_prompts
        # loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if optimizer_fc is not None:
            optimizer_fc.step()
            optimizer_fc.zero_grad()

        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            loss = reduce_value(loss, average=True)

        acc = compute_accuracy(output, label, topk=(1,))[0].item()

        loss_meter.update(loss.item(), image.shape[0])
        cls_loss_meter.update(loss_cls.item(), image.shape[0])
        prompts_loss_meter.update(loss_prompts.item(), image.shape[0])
        acc_meter.update(acc, 1)

        # compute time
        # torch.cuda.synchronize()
        batch_time.update(time.time() - start)

        # log lr
        wandb.log({'lr': model.get_current_lr()})

        meet_freq = iters % cfg.TRAIN.PRINT_FREQ == 0
        if meet_freq:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                nb_remain = tot_iter - iters
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"Iter [{iters}/{tot_iter}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})"]
                info += [f"cls_loss {cls_loss_meter.val:.3f} ({cls_loss_meter.avg:.3f})"]
                info += [f"prompts_loss {prompts_loss_meter.val:.3f} ({prompts_loss_meter.avg:.3f})"]
                info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    info += [f"lr {model.module.get_current_lr():.4e}"]
                else:
                    info += [f"lr {model.get_current_lr():.4e}"]
                if cfg.TRAINER.NAME == 'baseline_cattn_vocabloss_shembed_zsinit_lscale_lnable_wiseft_nxcattn':
                    info += [f"lscale {model.model.cls_head.logit_scale:.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))


            if cfg.TRAINER.NAME == 'baseline_cattn_vocabloss_shembed_zsinit_lscale_lnable_wiseft_nxcattn':
                wandb.log({'train loss': loss_meter.val,
                           'train acc': acc_meter.val,
                           'train cls loss': cls_loss_meter.val,
                           'train prompts loss': prompts_loss_meter.val,
                           'iter': iters,
                           'logit scale': model.model.cls_head.logit_scale.cpu().detach()
                           })
            else:
                wandb.log({'train loss': loss_meter.val,
                           'train acc': acc_meter.val,
                           'train cls loss': cls_loss_meter.val,
                           'train prompts loss': prompts_loss_meter.val,
                           'iter': iters
                           })

        # 2.meet epoch: save checkpoint
        sdir = cfg.OUTPUT_DIR
        if iters % cfg.TRAIN.SAVE_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.save_model(iters, sdir, is_best=False)
                else:
                    model.save_model(iters, sdir, is_best=False)

        # 3.meet epoch: do test
        ratio = 0.5
        if iters % cfg.TRAIN.TEST_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.set_model_mode("test")
                    results, results_wiseft, test_loss, test_wiseft_loss = test_wiseft(cfg, model, data,
                                                                                       ratio)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    results, results_wiseft, test_loss, test_wiseft_loss = test_wiseft(cfg, model, data,
                                                                                       ratio)
                    model.set_model_mode("train")
                wandb.log({'test acc': results["accuracy"],
                           f'test acc (wiseft_{ratio})': results_wiseft["accuracy"],
                           'test loss': test_loss,
                           f'test loss (wiseft_{ratio})': test_wiseft_loss})


def train_sweep(cfg, model, data, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim
    optimizer_fc = None
    try:
        optimizer_fc = model.optim_fc
    except:
        pass

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
    prompts_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    for epoch in range(1, tot_epoch+1):
        loss_meter.reset()
        prompts_loss_meter.reset()
        acc_meter.reset()
        batch_time.reset()

        # 3. train
        for batch_idx, batch in enumerate(data.train_loader):
            end = time.time()
            image, label = parse_batch(batch)
            # forward
            # output = model(image)
            # loss = criterion(output, label)
            output, loss_prompts = model(image, label)
            loss = criterion(output, label) + loss_prompts
            if not torch.isfinite(loss).all():
                logger.info(f"Loss is infinite or NaN! loss:{loss.item()}")
                time.sleep(2)
                break
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if optimizer_fc is not None:
                optimizer_fc.step()
                optimizer_fc.zero_grad()

            if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                loss = reduce_value(loss, average=True)

            acc = compute_accuracy(output, label, topk=(1,))[0].item()

            loss_meter.update(loss.item(), image.shape[0])
            prompts_loss_meter.update(loss_prompts.item(), image.shape[0])
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
                    info += [f"prompts_loss {prompts_loss_meter.val:.3f} ({prompts_loss_meter.avg:.3f})"]
                    info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                    if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                        info += [f"lr {model.module.get_current_lr():.4e}"]
                    else:
                        info += [f"lr {model.get_current_lr():.4e}"]
                    info += [f"eta {eta}"]
                    logger.info(" ".join(info))

        else:
            wandb.log({'train loss': loss_meter.avg,
                       'train acc': acc_meter.avg,
                       'train prompts loss': prompts_loss_meter.avg,
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
                        results, test_loss = test(cfg, model, data)
                        model.module.set_model_mode("train")
                    else:
                        model.set_model_mode("test")
                        results, test_loss = test(cfg, model, data)
                        model.set_model_mode("train")
                    wandb.log({'test acc': results["accuracy"],
                               'test loss': test_loss})
            continue
        # logger.info("Loss is infinite or NaN!")
        break


def train_lpclip(cfg, model, data, local_rank):
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
            # output = model(image, label)
            # loss = criterion(output, label)
            output = model(image, label)
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
                    results, test_loss = test_clip(cfg, model, data)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    results, test_loss = test_clip(cfg, model, data)
                    model.set_model_mode("train")
                wandb.log({'test acc': results["accuracy"],
                           'test loss': test_loss})


def test(cfg, model, data):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    evaluator = Classification(cfg, logger)
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()

    logger.info(f"Evaluate on the *test* set")

    for batch_idx, batch in enumerate(tqdm(data.test_loader)):
        with torch.no_grad():
            image, label = parse_batch(batch)
            output, _ = model(image)
            loss = criterion(output, label)
            evaluator.process(output, label)
            loss_meter.update(loss.item(), image.shape[0])

    return evaluator.evaluate(), loss_meter.avg     # results["accuracy"]


def test_wiseft(cfg, model, data, ratio=0.5):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    evaluator = Classification(cfg, logger)
    evaluator_wiseft = Classification(cfg, logger)
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    loss_wiseft_meter = AverageMeter()

    logger.info(f"Evaluate on the *test* set")
    head = deepcopy(model.model.cls_head.fc)
    zs_weights = deepcopy(model.model.zs_weights)       # check if need .clone(): no
    wiseft_weights = (1 - ratio) * head.weight.data + ratio * zs_weights
    model.model.wiseft_head.fc.weight.data = wiseft_weights
    for batch_idx, batch in enumerate(tqdm(data.test_loader)):
        with torch.no_grad():
            image, label = parse_batch(batch)
            logits, logits_wiseft = model(image)
            loss = criterion(logits, label)
            loss_wiseft = criterion(logits_wiseft, label)
            loss_meter.update(loss.item(), image.shape[0])
            loss_wiseft_meter.update(loss_wiseft.item(), image.shape[0])
            evaluator.process(logits, label)
            evaluator_wiseft.process(logits_wiseft, label)

    return evaluator.evaluate(), evaluator_wiseft.evaluate(), loss_meter.avg, loss_wiseft_meter.avg    # results["accuracy"]


def test_clip(cfg, model, data):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    evaluator = Classification(cfg, logger)
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()

    logger.info(f"Evaluate on the *test* set")

    for batch_idx, batch in enumerate(tqdm(data.test_loader)):
        with torch.no_grad():
            image, label = parse_batch(batch)
            output = model(image)
            loss = criterion(output, label)
            evaluator.process(output, label)
            loss_meter.update(loss.item(), image.shape[0])

    return evaluator.evaluate(), loss_meter.avg     # results["accuracy"]


def parse_batch(batch):
    input = batch["img"]
    label = batch["label"]
    input = input.to(device)
    label = label.to(device)
    return input, label


def train_wandb_two_stage(cfg, model, data, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim
    optimizer_fc = None
    try:
        optimizer_fc = model.optim_fc
    except:
        pass

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
    prompts_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    for epoch in range(1, tot_epoch + 1):
        if epoch <= cfg.TRAIN.FIX_EPOCH:
            model.model.cls_head.fc.weight.requires_grad_(False)
        else:
            model.model.cls_head.fc.weight.requires_grad_(True)
        loss_meter.reset()
        prompts_loss_meter.reset()
        acc_meter.reset()
        batch_time.reset()

        for _ in range(cfg.INPUT.NUM_VIEWS):
            # 3. train
            for batch_idx, batch in enumerate(data.train_loader):
                end = time.time()
                image, label = parse_batch(batch)
                # forward
                # output = model(image, label)
                # loss = criterion(output, label)
                output, loss_prompts = model(image, label)
                loss = criterion(output, label) + loss_prompts
                # loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if optimizer_fc is not None:
                    optimizer_fc.step()
                    optimizer_fc.zero_grad()

                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    loss = reduce_value(loss, average=True)

                acc = compute_accuracy(output, label, topk=(1,))[0].item()

                loss_meter.update(loss.item(), image.shape[0])
                prompts_loss_meter.update(loss_prompts.item(),image.shape[0])
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
                        info += [f"prompts_loss {prompts_loss_meter.val:.3f} ({prompts_loss_meter.avg:.3f})"]
                        info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                            info += [f"lr {model.module.get_current_lr():.4e}"]
                        else:
                            info += [f"lr {model.get_current_lr():.4e}"]
                        info += [f"eta {eta}"]
                        logger.info(" ".join(info))

        wandb.log({'train loss': loss_meter.avg,
                   'train acc': acc_meter.avg,
                   'train prompts loss': prompts_loss_meter.avg,
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
                    results, test_loss = test(cfg, model, data)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    results, test_loss = test(cfg, model, data)
                    model.set_model_mode("train")
                wandb.log({'test acc': results["accuracy"],
                           'test loss': test_loss})


def train_sweep_iter_wiseft(cfg, model, data, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim
    scheduler = model.sched
    optimizer_fc = None
    try:
        optimizer_fc = model.optim_fc
    except:
        pass

    if device:
        model.to(local_rank)
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    tot_iter = cfg.OPTIM.MAX_ITER

    # 1. loss
    criterion = nn.CrossEntropyLoss()

    # 2. meter()
    loss_meter = AverageMeter()
    prompts_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    image_loader_iter = iter(data.train_loader)

    for iters in range(1, tot_iter + 1):
        start = time.time()
        # update lr
        scheduler.step()
        try:
            image, label = parse_batch(next(image_loader_iter))
        except StopIteration:
            image_loader_iter = iter(data.train_loader)
            image, label = parse_batch(next(image_loader_iter))

        output, loss_prompts = model(image, label)
        loss = criterion(output, label) + loss_prompts
        if not torch.isfinite(loss).all():
            logger.info(f"Loss is infinite or NaN! loss:{loss.item()}")
            time.sleep(2)
            break

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if optimizer_fc is not None:
            optimizer_fc.step()
            optimizer_fc.zero_grad()

        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            loss = reduce_value(loss, average=True)

        acc = compute_accuracy(output, label, topk=(1,))[0].item()

        loss_meter.update(loss.item(), image.shape[0])
        prompts_loss_meter.update(loss_prompts.item(), image.shape[0])
        acc_meter.update(acc, 1)

        batch_time.update(time.time() - start)

        # log lr
        wandb.log({'lr': model.get_current_lr()})

        meet_freq = iters % cfg.TRAIN.PRINT_FREQ == 0
        if meet_freq:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                nb_remain = tot_iter - iters
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"Iter [{iters}/{tot_iter}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})"]
                info += [f"prompts_loss {prompts_loss_meter.val:.3f} ({prompts_loss_meter.avg:.3f})"]
                info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    info += [f"lr {model.module.get_current_lr():.4e}"]
                else:
                    info += [f"lr {model.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))

            wandb.log({'train loss': loss_meter.val,
                       'train acc': acc_meter.val,
                       'train prompts loss': prompts_loss_meter.val,
                       'iter': iters
                       })

        # 2.meet epoch: save checkpoint
        sdir = cfg.OUTPUT_DIR
        if iters % cfg.TRAIN.SAVE_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.save_model(iters, sdir, is_best=False)
                else:
                    model.save_model(iters, sdir, is_best=False)

        # 3.meet epoch: do test
        ratio = 0.5
        if iters % cfg.TRAIN.TEST_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.set_model_mode("test")
                    results, results_wiseft, test_loss, test_wiseft_loss = test_wiseft(cfg, model, data,
                                                                                       ratio)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    results, results_wiseft, test_loss, test_wiseft_loss = test_wiseft(cfg, model, data,
                                                                                       ratio)
                    model.set_model_mode("train")
                wandb.log({'test acc': results["accuracy"],
                           f'test acc (wiseft_{ratio})': results_wiseft["accuracy"],
                           'test loss': test_loss,
                           f'test loss (wiseft_{ratio})': test_wiseft_loss})


# def train_wandb_iter_wiseft_val(cfg, model, data, local_rank):
def train_wandb_iter_wiseft_val(cfg, model, data, image_loader,
                                val_loader, test_loader, output_dir, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim
    scheduler = model.sched
    optimizer_fc = None
    try:
        optimizer_fc = model.optim_fc
    except:
        pass

    if device:
        model.to(local_rank)
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    tot_iter = cfg.OPTIM.MAX_ITER

    # 1. loss
    criterion = nn.CrossEntropyLoss()

    # 2. meter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    prompts_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    image_loader_iter = iter(image_loader)
    best_val_dict = {
        "iter": 1,
        "val_acc": 0.,
        "model": None
    }

    test_freq = cfg.TRAIN.TEST_FREQ
    for iters in range(1, tot_iter+1):
        start = time.time()
        # update lr
        scheduler.step()
        try:
            image, label = parse_batch(next(image_loader_iter))
        except StopIteration:
            image_loader_iter = iter(image_loader)
            image, label = parse_batch(next(image_loader_iter))

        output, loss_prompts = model(image, label)
        loss = criterion(output, label) + loss_prompts
        loss_cls = loss - loss_prompts
        # loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if optimizer_fc is not None:
            optimizer_fc.step()
            optimizer_fc.zero_grad()

        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            loss = reduce_value(loss, average=True)

        acc = compute_accuracy(output, label, topk=(1,))[0].item()

        loss_meter.update(loss.item(), image.shape[0])
        cls_loss_meter.update(loss_cls.item(), image.shape[0])
        prompts_loss_meter.update(loss_prompts.item(), image.shape[0])
        acc_meter.update(acc, 1)

        # compute time
        # torch.cuda.synchronize()
        batch_time.update(time.time() - start)

        # log lr
        wandb.log({'lr': model.get_current_lr()})

        meet_freq = iters % cfg.TRAIN.PRINT_FREQ == 0
        if meet_freq:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                nb_remain = tot_iter - iters
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"Iter [{iters}/{tot_iter}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})"]
                info += [f"cls_loss {cls_loss_meter.val:.3f} ({cls_loss_meter.avg:.3f})"]
                info += [f"prompts_loss {prompts_loss_meter.val:.3f} ({prompts_loss_meter.avg:.3f})"]
                info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    info += [f"lr {model.module.get_current_lr():.4e}"]
                else:
                    info += [f"lr {model.get_current_lr():.4e}"]
                if cfg.TRAINER.NAME == 'baseline_cattn_vocabloss_shembed_zsinit_lscale_lnable_wiseft_nxcattn':
                    info += [f"lscale {model.model.cls_head.logit_scale:.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))


            if cfg.TRAINER.NAME == 'baseline_cattn_vocabloss_shembed_zsinit_lscale_lnable_wiseft_nxcattn':
                wandb.log({'train loss': loss_meter.val,
                           'train acc': acc_meter.val,
                           'train cls loss': cls_loss_meter.val,
                           'train prompts loss': prompts_loss_meter.val,
                           'iter': iters,
                           'logit scale': model.model.cls_head.logit_scale.cpu().detach()
                           })
            else:
                wandb.log({'train loss': loss_meter.val,
                           'train acc': acc_meter.val,
                           'train cls loss': cls_loss_meter.val,
                           'train prompts loss': prompts_loss_meter.val,
                           'iter': iters
                           })

        # 2.meet epoch: save checkpoint
        sdir = output_dir
        if iters % cfg.TRAIN.SAVE_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.save_model(iters, sdir, is_best=False)
                else:
                    model.save_model(iters, sdir, is_best=False)

        # 3.meet epoch: do test
        # NOTE: change test_loader -> val_loader
        if iters >= (cfg.OPTIM.MAX_ITER // 2):
            test_freq = cfg.TRAIN.TEST_FREQ * 4
        if (iters % test_freq == 0) or iters == 1:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.set_model_mode("test")
                    results, test_loss = val_wiseft_head(cfg, model, val_loader)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    results, test_loss = val_wiseft_head(cfg, model, val_loader)
                    model.set_model_mode("train")
                if results["accuracy"] > best_val_dict["val_acc"]:
                    best_val_dict["iter"] = iters
                    best_val_dict["val_acc"] = results["accuracy"]
                    best_val_dict["model"] = deepcopy(model.state_dict())

                wandb.log({'val acc': results["accuracy"],
                           'val loss': test_loss,
                           'best val iter': best_val_dict["iter"]})

                wandb.log({'test acc': 0.,
                           'test acc (wiseft_0.5)': 0.,
                           'test acc (wiseft_1.0)': 0.,
                           'test loss': 0.,
                           'test loss (wiseft_0.5)': 0.,
                           'test loss (wiseft_1.0)': 0.})

    # final: test using the best val model
    model.load_state_dict(best_val_dict["model"])
    ratio = 0.5
    if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
        model.module.set_model_mode("test")
        results, results_wiseft, results_wiseft2, test_loss, test_wiseft_loss, test_wiseft_loss2 = test_wiseft_val(cfg, model, test_loader, ratio)
        model.module.set_model_mode("train")
    else:
        model.set_model_mode("test")
        results, results_wiseft, results_wiseft2, test_loss, test_wiseft_loss, test_wiseft_loss2 = test_wiseft_val(cfg, model, test_loader, ratio)
        model.set_model_mode("train")

    test_results = {
        'test acc': results["accuracy"],
        f'test acc (wiseft_{ratio})': results_wiseft["accuracy"],
        'test acc (wiseft_1.0)': results_wiseft2["accuracy"],
        'test loss': test_loss,
        f'test loss (wiseft_{ratio})': test_wiseft_loss,
        'test loss (wiseft_1.0)': test_wiseft_loss2
    }
    wandb.log(test_results)

    # save the test results
    test_path = os.path.join(output_dir, "test.json")
    with open(test_path, 'w') as f:
        json.dump(test_results, f)

    # save the best model
    sdir = output_dir
    model.save_model(0, sdir, is_best=True)


# using wiseft head to do early stopping
def val_wiseft_head(cfg, model, val_loader):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    evaluator = Classification(cfg, logger)
    evaluator_wiseft = Classification(cfg, logger)
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    loss_wiseft_meter = AverageMeter()

    ratio=0.5
    logger.info(f"Evaluate on the *val* set")
    head = deepcopy(model.model.cls_head.fc)
    zs_weights = deepcopy(model.model.zs_weights)  # check if need .clone(): no
    wiseft_weights = (1 - ratio) * head.weight.data + ratio * zs_weights
    model.model.wiseft_head.fc.weight.data = wiseft_weights
    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            image, label = parse_batch(batch)
            logits, logits_wiseft, _ = model(image)
            loss = criterion(logits, label)
            loss_wiseft = criterion(logits_wiseft, label)
            loss_meter.update(loss.item(), image.shape[0])
            loss_wiseft_meter.update(loss_wiseft.item(), image.shape[0])
            evaluator.process(logits, label)
            evaluator_wiseft.process(logits_wiseft, label)

    # return evaluator.evaluate(), loss_meter.avg
    return evaluator_wiseft.evaluate(), loss_wiseft_meter.avg


# using head to do early stopping
def val_head(cfg, model, val_loader):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    evaluator = Classification(cfg, logger)
    evaluator_wiseft = Classification(cfg, logger)
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    loss_wiseft_meter = AverageMeter()

    ratio=0.5
    logger.info(f"Evaluate on the *val* set")
    head = deepcopy(model.model.cls_head.fc)
    zs_weights = deepcopy(model.model.zs_weights)  # check if need .clone(): no
    wiseft_weights = (1 - ratio) * head.weight.data + ratio * zs_weights
    model.model.wiseft_head.fc.weight.data = wiseft_weights
    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            image, label = parse_batch(batch)
            logits, logits_wiseft, _ = model(image)
            loss = criterion(logits, label)
            loss_wiseft = criterion(logits_wiseft, label)
            loss_meter.update(loss.item(), image.shape[0])
            loss_wiseft_meter.update(loss_wiseft.item(), image.shape[0])
            evaluator.process(logits, label)
            evaluator_wiseft.process(logits_wiseft, label)

    return evaluator.evaluate(), loss_meter.avg
    # return evaluator_wiseft.evaluate(), loss_wiseft_meter.avg


def test_wiseft_val(cfg, model, test_loader, ratio=0.5):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    evaluator = Classification(cfg, logger)
    evaluator_wiseft = Classification(cfg, logger)
    evaluator_wiseft2 = Classification(cfg, logger)
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    loss_wiseft_meter = AverageMeter()
    loss_wiseft_meter2 = AverageMeter()

    logger.info(f"Evaluate on the *test* set")
    head = deepcopy(model.model.cls_head.fc)
    zs_weights = deepcopy(model.model.zs_weights)       # check if need .clone(): no
    wiseft_weights = (1 - ratio) * head.weight.data + ratio * zs_weights
    model.model.wiseft_head.fc.weight.data = wiseft_weights
    model.model.wiseft_head2.fc.weight.data = zs_weights
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            image, label = parse_batch(batch)
            logits, logits_wiseft, logits_wiseft2 = model(image)
            loss = criterion(logits, label)
            loss_wiseft = criterion(logits_wiseft, label)
            loss_wiseft2 = criterion(logits_wiseft2, label)
            loss_meter.update(loss.item(), image.shape[0])
            loss_wiseft_meter.update(loss_wiseft.item(), image.shape[0])
            loss_wiseft_meter2.update(loss_wiseft2.item(), image.shape[0])
            evaluator.process(logits, label)
            evaluator_wiseft.process(logits_wiseft, label)
            evaluator_wiseft2.process(logits_wiseft2, label)

    return evaluator.evaluate(), evaluator_wiseft.evaluate(), evaluator_wiseft2.evaluate(),\
           loss_meter.avg, loss_wiseft_meter.avg, loss_wiseft_meter2.avg


# train_wandb_iter_wiseft_val
def train_caption(cfg, model, data, image_loader, val_loader, test_loader, output_dir, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer = model.optim
    scheduler = model.sched
    if cfg.OPTIM.LORA_OPTIM:
        optimizer_lora = model.optim_lora
        scheduler_lora = model.sched_lora
    else:
        optimizer_lora = None
        scheduler_lora = None

    if device:
        model.to(local_rank)
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    tot_iter = cfg.OPTIM.MAX_ITER

    # 1. loss
    criterion = nn.CrossEntropyLoss()

    # 2. meter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    category_loss_meter = AverageMeter()
    instance_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    image_loader_iter = iter(image_loader)
    best_val_dict = {
        "iter": 1,
        "val_acc": 0.,
        "model": None
    }

    test_freq = cfg.TRAIN.TEST_FREQ
    for iters in range(1, tot_iter+1):
        model.set_model_mode("train")
        start = time.time()
        # update lr
        scheduler.step()
        if scheduler_lora is not None:
            scheduler_lora.step()
        try:
            image, label, caption = parse_batch_caption(next(image_loader_iter))
        except StopIteration:
            image_loader_iter = iter(image_loader)
            image, label, caption = parse_batch_caption(next(image_loader_iter))

        if "auxi" in cfg.TRAINER.NAME:
            output, loss_category, loss_instance, loss_final = model(image, label, caption)
        else:
            output, loss_category, loss_instance = model(image, label, caption)
        loss_prompts = loss_category + loss_instance
        loss = criterion(output, label) + loss_prompts
        loss_cls = loss - loss_prompts
        # loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if optimizer_lora is not None:
            optimizer_lora.step()
            optimizer_lora.zero_grad()

        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            loss = reduce_value(loss, average=True)

        acc = compute_accuracy(output, label, topk=(1,))[0].item()

        loss_meter.update(loss.item(), image.shape[0])
        cls_loss_meter.update(loss_cls.item(), image.shape[0])
        category_loss_meter.update(loss_category.item(), image.shape[0])
        if "auxi" in cfg.TRAINER.NAME:
            instance_loss_meter.update(loss_final.item(), image.shape[0])
        else:
            instance_loss_meter.update(loss_instance.item(), image.shape[0])
        acc_meter.update(acc, 1)

        # compute time
        # torch.cuda.synchronize()
        batch_time.update(time.time() - start)

        # log lr

        if optimizer_lora is not None:
            wandb.log({'lr': model.get_current_lr(),
                       'lr_lora': model.get_specific_lr('clip_model')
                       })
        else:
            wandb.log({'lr': model.get_current_lr()})

        meet_freq = iters % cfg.TRAIN.PRINT_FREQ == 0
        if meet_freq:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                nb_remain = tot_iter - iters
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"Iter [{iters}/{tot_iter}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})"]
                info += [f"cls_loss {cls_loss_meter.val:.3f} ({cls_loss_meter.avg:.3f})"]
                info += [f"category_loss {category_loss_meter.val:.3f} ({category_loss_meter.avg:.3f})"]
                info += [f"instance_loss {instance_loss_meter.val:.3f} ({instance_loss_meter.avg:.3f})"]
                info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    info += [f"lr {model.module.get_current_lr():.4e}"]
                else:
                    info += [f"lr {model.get_current_lr():.4e}"]
                if cfg.TRAINER.NAME == 'baseline_cattn_vocabloss_shembed_zsinit_lscale_lnable_wiseft_nxcattn':
                    info += [f"lscale {model.model.cls_head.logit_scale:.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))


            wandb.log({'train loss': loss_meter.val,
                       'train acc': acc_meter.val,
                       'train cls loss': cls_loss_meter.val,
                       'train category loss': category_loss_meter.val,
                       'train instance loss': instance_loss_meter.val,
                       'iter': iters
                       })

        # 2.meet epoch: save checkpoint
        sdir = output_dir
        if iters % cfg.TRAIN.SAVE_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.save_model(iters, sdir, is_best=False)
                else:
                    model.save_model(iters, sdir, is_best=False)

        # if iters >= (cfg.OPTIM.MAX_ITER // 2):
        #     test_freq = cfg.TRAIN.TEST_FREQ * 4

        # 3.meet epoch: do test
        if (iters % cfg.TRAIN.TEST_FREQ == 0) or iters == 1:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.set_model_mode("test")
                    if cfg.TRAIN.VAL_WISEFT:
                        results, test_loss = val_wiseft_head(cfg, model, val_loader)
                    else:
                        results, test_loss = val_head(cfg, model, val_loader)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    if cfg.TRAIN.VAL_WISEFT:
                        results, test_loss = val_wiseft_head(cfg, model, val_loader)
                    else:
                        results, test_loss = val_head(cfg, model, val_loader)
                    model.set_model_mode("train")
                if results["accuracy"] > best_val_dict["val_acc"]:
                    best_val_dict["iter"] = iters
                    best_val_dict["val_acc"] = results["accuracy"]
                    best_val_dict["model"] = deepcopy(model.state_dict())

                wandb.log({'val acc': results["accuracy"],
                           'val loss': test_loss,
                           'best val iter': best_val_dict["iter"]})

                wandb.log({'test acc': 0.,
                           'test acc (wiseft_0.5)': 0.,
                           'test acc (wiseft_1.0)': 0.,
                           'test loss': 0.,
                           'test loss (wiseft_0.5)': 0.,
                           'test loss (wiseft_1.0)': 0.})

                # enabled = set()
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         enabled.add(name)
                # logger.info(f"Parameters to be updated: {enabled}")

    # final: test using the best val model
    model.load_state_dict(best_val_dict["model"])
    ratio = 0.5
    if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
        model.module.set_model_mode("test")
        results, results_wiseft, results_wiseft2, test_loss, test_wiseft_loss, test_wiseft_loss2 = test_wiseft_val(cfg, model, test_loader, ratio)
        model.module.set_model_mode("train")
    else:
        model.set_model_mode("test")
        results, results_wiseft, results_wiseft2, test_loss, test_wiseft_loss, test_wiseft_loss2 = test_wiseft_val(cfg, model, test_loader, ratio)
        model.set_model_mode("train")

    test_results = {
         'test acc': results["accuracy"],
         f'test acc (wiseft_{ratio})': results_wiseft["accuracy"],
         'test acc (wiseft_1.0)': results_wiseft2["accuracy"],
         'test loss': test_loss,
         f'test loss (wiseft_{ratio})': test_wiseft_loss,
         'test loss (wiseft_1.0)': test_wiseft_loss2
    }
    wandb.log(test_results)

    # save the test results
    test_path = os.path.join(output_dir, "test.json")
    with open(test_path, 'w') as f:
        json.dump(test_results, f)

    # # save the best model
    # sdir = output_dir
    # model.save_model(0, sdir, is_best=True)

def inference(cfg, model, data, image_loader, val_loader, test_loader, output_dir, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)

    if device:
        model.to(local_rank)
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    ratio = 0.5
    if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
        model.module.set_model_mode("test")
        results, results_wiseft, results_wiseft2, test_loss, test_wiseft_loss, test_wiseft_loss2 = test_wiseft_val(cfg, model, test_loader, ratio)
        model.module.set_model_mode("train")
    else:
        model.set_model_mode("test")
        results, results_wiseft, results_wiseft2, test_loss, test_wiseft_loss, test_wiseft_loss2 = test_wiseft_val(cfg, model, test_loader, ratio)
        model.set_model_mode("train")

    test_results = {
        'test acc': results["accuracy"],
        f'test acc (wiseft_{ratio})': results_wiseft["accuracy"],
        'test acc (wiseft_1.0)': results_wiseft2["accuracy"],
        'test loss': test_loss,
        f'test loss (wiseft_{ratio})': test_wiseft_loss,
        'test loss (wiseft_1.0)': test_wiseft_loss2
    }
    wandb.log(test_results)

    # save the test results
    test_path = os.path.join(output_dir, "test.json")
    with open(test_path, 'w') as f:
        json.dump(test_results, f)


def parse_batch_caption(batch):
    input = batch["img"]
    label = batch["label"]
    caption = batch['tokenized_caption']
    input = input.to(device)
    label = label.to(device)
    caption = caption.to(device)
    return input, label, caption


# train_wandb_iter_wiseft_val_two_stage
def train_wandb_iter_wiseft_val_fixedfirst(cfg, model, data, image_loader,
                                val_loader, test_loader, output_dir, local_rank):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    optimizer_bonder = model.optim_bonder
    scheduler_bonder = model.sched_bonder
    optimizer_fc = model.optim_fc
    scheduler_fc = model.sched_fc
    optimizer_lora = model.optim_lora
    scheduler_lora = model.sched_lora

    if device:
        model.to(local_rank)
        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs for training".format(torch.cuda.device_count()))
            # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    tot_iter = cfg.OPTIM.MAX_ITER

    # 1. loss
    criterion = nn.CrossEntropyLoss()

    # 2. meter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    category_loss_meter = AverageMeter()
    instance_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = GradScaler()

    image_loader_iter = iter(image_loader)
    best_val_dict = {
        "iter": 1,
        "val_acc": 0.,
        "model": None
    }

    for iters in range(1, tot_iter+1):
        if iters <= cfg.TRAIN.FIX_EPOCH:
            model.model.cls_head.fc.weight.requires_grad_(False)
        else:
            model.model.cls_head.fc.weight.requires_grad_(True)
            scheduler_fc.step()
            scheduler_lora.step()
        scheduler_bonder.step()     # NOTE second stage: bonder also training

        start = time.time()
        try:
            image, label, caption = parse_batch_caption(next(image_loader_iter))
        except StopIteration:
            image_loader_iter = iter(image_loader)
            image, label, caption = parse_batch_caption(next(image_loader_iter))

        output, loss_category, loss_instance = model(image, label, caption)
        loss_prompts = loss_category + loss_instance
        loss = criterion(output, label) + loss_prompts
        loss_cls = loss - loss_prompts
        # loss = criterion(output, label)
        loss.backward()
        if iters > cfg.TRAIN.FIX_EPOCH:
            optimizer_fc.step()
            optimizer_fc.zero_grad()
            optimizer_lora.step()
            optimizer_lora.zero_grad()
        optimizer_bonder.step()
        optimizer_bonder.zero_grad()

        if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
            loss = reduce_value(loss, average=True)

        acc = compute_accuracy(output, label, topk=(1,))[0].item()

        loss_meter.update(loss.item(), image.shape[0])
        cls_loss_meter.update(loss_cls.item(), image.shape[0])
        category_loss_meter.update(loss_category.item(), image.shape[0])
        instance_loss_meter.update(loss_instance.item(), image.shape[0])
        acc_meter.update(acc, 1)

        # compute time
        # torch.cuda.synchronize()
        batch_time.update(time.time() - start)

        # log lr
        wandb.log({'lr': model.get_current_lr(),
                   'lr_fc': model.get_specific_lr('cls_head')})

        meet_freq = iters % cfg.TRAIN.PRINT_FREQ == 0
        if meet_freq:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                nb_remain = tot_iter - iters
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"Iter [{iters}/{tot_iter}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})"]
                info += [f"cls_loss {cls_loss_meter.val:.3f} ({cls_loss_meter.avg:.3f})"]
                info += [f"category_loss {category_loss_meter.val:.3f} ({category_loss_meter.avg:.3f})"]
                info += [f"instance_loss {instance_loss_meter.val:.3f} ({instance_loss_meter.avg:.3f})"]
                info += [f"acc {acc_meter.val:.3f} ({acc_meter.avg:.3f})"]
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    info += [f"lr {model.module.get_current_lr():.4e}"]
                else:
                    info += [f"lr {model.get_current_lr():.4e}"]
                if cfg.TRAINER.NAME == 'baseline_cattn_vocabloss_shembed_zsinit_lscale_lnable_wiseft_nxcattn':
                    info += [f"lscale {model.model.cls_head.logit_scale:.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))

            wandb.log({'train loss': loss_meter.val,
                       'train acc': acc_meter.val,
                       'train cls loss': cls_loss_meter.val,
                       'train category loss': category_loss_meter.val,
                       'train instance loss': instance_loss_meter.val,
                       'iter': iters
                       })

        # 2.meet epoch: save checkpoint
        sdir = output_dir
        if iters % cfg.TRAIN.SAVE_FREQ == 0:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.save_model(iters, sdir, is_best=False)
                else:
                    model.save_model(iters, sdir, is_best=False)

        # 3.meet epoch: do test
        if (iters % cfg.TRAIN.TEST_FREQ == 0) or iters == 1:
            if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
                pass
            else:
                if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
                    model.module.set_model_mode("test")
                    if cfg.TRAIN.VAL_WISEFT:
                        results, test_loss = val_wiseft_head(cfg, model, val_loader)
                    else:
                        results, test_loss = val_head(cfg, model, val_loader)
                    model.module.set_model_mode("train")
                else:
                    model.set_model_mode("test")
                    if cfg.TRAIN.VAL_WISEFT:
                        results, test_loss = val_wiseft_head(cfg, model, val_loader)
                    else:
                        results, test_loss = val_head(cfg, model, val_loader)
                    model.set_model_mode("train")
                if results["accuracy"] > best_val_dict["val_acc"]:
                    best_val_dict["iter"] = iters
                    best_val_dict["val_acc"] = results["accuracy"]
                    best_val_dict["model"] = deepcopy(model.state_dict())

                wandb.log({'val acc': results["accuracy"],
                           'val loss': test_loss,
                           'best val iter': best_val_dict["iter"]})

                wandb.log({'test acc': 0.,
                           'test acc (wiseft_0.5)': 0.,
                           'test acc (wiseft_1.0)': 0.,
                           'test loss': 0.,
                           'test loss (wiseft_0.5)': 0.,
                           'test loss (wiseft_1.0)': 0.})

    # final: test using the best val model
    model.load_state_dict(best_val_dict["model"])
    ratio = 0.5
    if cfg.TRAIN.DIST_TRAIN and torch.cuda.device_count() > 1:
        model.module.set_model_mode("test")
        results, results_wiseft, results_wiseft2, test_loss, test_wiseft_loss, test_wiseft_loss2 = test_wiseft_val(cfg, model, test_loader, ratio)
        model.module.set_model_mode("train")
    else:
        model.set_model_mode("test")
        results, results_wiseft, results_wiseft2, test_loss, test_wiseft_loss, test_wiseft_loss2 = test_wiseft_val(cfg, model, test_loader, ratio)
        model.set_model_mode("train")

    test_results = {
        'test acc': results["accuracy"],
        f'test acc (wiseft_{ratio})': results_wiseft["accuracy"],
        'test acc (wiseft_1.0)': results_wiseft2["accuracy"],
        'test loss': test_loss,
        f'test loss (wiseft_{ratio})': test_wiseft_loss,
        'test loss (wiseft_1.0)': test_wiseft_loss2
    }
    wandb.log(test_results)

    # save the test results
    test_path = os.path.join(output_dir, "test.json")
    with open(test_path, 'w') as f:
        json.dump(test_results, f)

    # save the best model
    sdir = output_dir
    model.save_model(0, sdir, is_best=True)