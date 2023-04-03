import argparse
import logging
import torch
import torch.distributed as dist

# from dassl.utils import setup_logger, set_random_seed, collect_env_info
# from dassl.config import get_cfg_default
# from dassl.engine import build_trainer

from models import Baseline, lpclip, Baseline_cattn, Baseline_cattn_vocabloss, Baseline_cattn_vocabloss_wotextloss, Baseline_cattn_vocabloss_cpvocab, Baseline_cattn_vocabloss_shembed, Baseline_cattn_vocabloss_shembed_mul
from configs import get_cfg_default
from datasets import DataManager
from processor import train, train_sweep
from tools.utils import set_random_seed, collect_env_info
from tools.logger import setup_logger

import wandb
import warnings
warnings.filterwarnings("ignore")


_MODEL = {
    'baseline': Baseline,
    'baseline_cattn': Baseline_cattn,
    'baseline_cattn_vocabloss': Baseline_cattn_vocabloss,
    'baseline_cattn_vocabloss_wotextloss': Baseline_cattn_vocabloss_wotextloss,
    'baseline_cattn_vocabloss_cpvocab': Baseline_cattn_vocabloss_cpvocab,
    'baseline_cattn_vocabloss_shembed': Baseline_cattn_vocabloss_shembed,
    'baseline_cattn_vocabloss_shembed_mul': Baseline_cattn_vocabloss_shembed_mul,
    'lpclip': lpclip
}


def print_args(args, cfg):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    logger.info("***************")
    logger.info("** Arguments **")
    logger.info("***************")
    optkeys = list(args.keys())
    optkeys.sort()
    for key in optkeys:
        logger.info("{}: {}".format(key, args[key]))
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)


def reset_cfg(cfg, args):
    if args['root']:
        cfg.DATASET.ROOT = args['root']

    if args['output_dir']:
        cfg.OUTPUT_DIR = args['output_dir']

    if args['resume']:
        cfg.RESUME = args['resume']

    if args['seed']:
        cfg.SEED = args['seed']

    if args['source_domains']:
        cfg.DATASET.SOURCE_DOMAINS = args['source_domains']

    if args['target_domains']:
        cfg.DATASET.TARGET_DOMAINS = args['target_domains']

    if args['transforms']:
        cfg.INPUT.TRANSFORMS = args['transforms']

    # if args.trainer:
    #     cfg.TRAINER.NAME = args.trainer

    if args['backbone']:
        cfg.MODEL.BACKBONE.NAME = args['backbone']

    if args['head']:
        cfg.MODEL.HEAD.NAME = args['head']

    if args['dist_train']:
        cfg.TRAIN.DIST_TRAIN = args['dist_train']


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.BASELINE = CN()
    cfg.TRAINER.BASELINE.N_CTX = 16  # number of context vectors
    cfg.TRAINER.BASELINE.CTX_INIT = ""  # initialization words
    cfg.TRAINER.BASELINE.FUSE = "cat"
    cfg.TRAINER.BASELINE.FEA_SCALE = 1.
    cfg.TRAINER.BASELINE.LOG_SCALE = 1.

    cfg.TRAIN.TEST_FREQ = 5
    cfg.TRAIN.SAVE_FREQ = 20
    cfg.TRAIN.DIST_TRAIN = False
    cfg.TRAINER.PREC = 'fp16'


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args['dataset_config_file']:
        cfg.merge_from_file(args['dataset_config_file'])

    # 2. From the method config file
    if args['config_file']:
        cfg.merge_from_file(args['config_file'])

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args['opts'])

    # cfg.freeze()

    return cfg


def main(config=None):
    with wandb.init(config=config):

        config = wandb.config
        args = config.args

        cfg = setup_cfg(args)

        logger = logging.getLogger(cfg.TRAINER.NAME)
        # update cfg with sweep config
        cfg.OPTIM.NAME = wandb.config.optim
        cfg.OPTIM.LR = wandb.config.lr
        cfg.OPTIM.WARMUP_CONS_LR = wandb.config.warmup_lr
        cfg.OPTIM.MAX_EPOCH = wandb.config.epoch
        cfg.OPTIM.WEIGHT_DECAY = wandb.config.weight_decay
        cfg.TRAINER.BASELINE.FEA_SCALE = wandb.config.fea_scale
        cfg.TRAINER.BASELINE.LOG_SCALE = wandb.config.log_scale
        cfg.freeze()
        wandb.run.name = 'vitb16-' + cfg.DATASET.NAME + f'-{cfg.DATASET.NUM_SHOTS}s-{cfg.TRAINER.NAME}-lscale{cfg.TRAINER.BASELINE.LOG_SCALE}-{cfg.OPTIM.NAME}-lr{cfg.OPTIM.LR}-e{cfg.OPTIM.MAX_EPOCH}'
        # wandb.run.name = 'cj_gb-' + cfg.DATASET.NAME + f'-{cfg.DATASET.NUM_SHOTS}s-{cfg.OPTIM.NAME}-lr{cfg.OPTIM.LR}'

        if cfg.SEED >= 0:
            logger.info("Setting fixed seed: {}".format(cfg.SEED))
            set_random_seed(cfg.SEED)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            torch.backends.cudnn.benchmark = True

        if cfg.TRAIN.DIST_TRAIN:
            torch.cuda.set_device(args['local_rank'])
            dist.init_process_group(backend='nccl', init_method='env://')

        if cfg.TRAIN.DIST_TRAIN and dist.get_rank() != 0:
            pass
        else:
            print_args(args, cfg)
            logger.info("Collecting env info ...")
            logger.info("** System info **\n{}\n".format(collect_env_info()))

        # 1.dataset
        data = DataManager(cfg)

        # 2.model ( +optim +sche)
        try:
            model = _MODEL[cfg.TRAINER.NAME](cfg, data.dataset.classnames)
        except:
            raise TypeError(f"Trainer {cfg.TRAINER.NAME} is not available.")

        # 3.train
        train_sweep(cfg, model, data, args['local_rank'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--dist-train", type=bool, default=False, help="path to config file"
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="baseline", help="name of trainer")  # CoOp
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    logger = setup_logger('baseline_cattn_vocabloss_shembed_mul', args.output_dir, if_train=True)    # todo: modify manually

    # Define the search space
    sweep_configuration = {
        'method': 'grid',
        'metric': {'goal': 'maximize', 'name': 'test acc'},
        'parameters':
            {
                # use 'values' for sweeping, 'value' for fixed
                'optim': {'value': 'adamw'},
                # 'lr': {'values': [0.006, 0.004, 0.002]},
                # 'lr': {'values': [0.1, 0.01]},
                'lr': {'value': 0.001},
                'weight_decay': {'value': 5e-4},
                'warmup_lr': {'value': 1e-5},
                # 'epoch': {'values': [50, 200]},
                'epoch': {'value': 50},
                # 'fea_scale': {'values': [64, 32, 16, 4]},
                'fea_scale': {'value': 1},
                'log_scale': {'values': [100, 64, 32, 16, 4]},
                'args': {'value': args}
            }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='baseline_cattn(_vocabloss)_sweep')

    # run = wandb.init()
    # run.name = f'test-1'
    # run = wandb.init(project='TransPAR-SaAttGCNTrans-v2-std', reinit=True)
    # run.name = f'18-peta-softmax-mask'
    # from pprint import pprint\\\
    #
    # pprint(wandb.config)
    # dic = vars(args)
    # wandb.config.args = dic

    wandb.agent(sweep_id, function=main, count=5)
