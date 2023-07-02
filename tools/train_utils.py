import sys
sys.path.insert(0, '.')

from models import Baseline, lpclip, Baseline_cattn, Baseline_cattn_vocabloss, \
    Baseline_cattn_vocabloss_wotextloss, Baseline_cattn_vocabloss_cpvocab, \
    Baseline_cattn_vocabloss_shembed, Baseline_cattn_vocabloss_shembed_mul, \
    Baseline_cattn_vocabloss_shembed_zsinit, Baseline_cattn_vocabloss_shembed_zsinit_fixed, \
    Baseline_cattn_vocabloss_shembed_lscale, Baseline_cattn_vocabloss_shembed_zsinit_optimfc, \
    Baseline_sam, lpsam, Baseline_cattn_embedloss, Baseline_cattn_vocabloss_shembed_zsinit_2xcattn, \
    Baseline_cattn_vocabloss_shembed_zsinit_2xcattn_pe, Baseline_cattn_vl_pd, \
    Baseline_cattn_vocabloss_shembed_zsinit_fixedfirst, Baseline_cattn_vocabloss_shembed_zsinit_textaug, \
    Baseline_cattn_vocabloss_shembed_zsinit_lscale, Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft, \
    Baseline_cattn_vocabloss_shembed_zsinit_mul_lscale_wiseft, Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_nxcattn, \
    Baseline_cattn_vocabloss_shembed_zsinit_lscale_lnable_wiseft_nxcattn, Baseline_cattn_coophead, \
    Baseline_cattn_vocabloss_shembed_zsinit_lscale_fixed, Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_wotextloss, \
    Baseline_sattn, Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_idfc, Baseline_cattn_wiseft_template_ensemble, \
    Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_add, Baseline_caption_wiseft, Baseline_caption_wiseft_multi_stream, \
    Baseline_abla_caption_only, Baseline_abla_multi_wo_caption, Baseline_caption_wiseft_nxcattn, Baseline_caption_wiseft_nxcattn_add, \
    Baseline_caption_wiseft_nxcattn_auxi, Baseline_caption_wiseft_multi_stream_projector, Baseline_caption_wiseft_multi_stream_bert, \
    Baseline_caption_wiseft_multi_stream_t5, Baseline_caption_wiseft_multi_stream_rn, Baseline_caption_wiseft_lora, \
    Baseline_caption_wiseft_lora_fixedfirst, Baseline_caption_wiseft_se_pre_all, Baseline_caption_wiseft_se_post, \
    Baseline_caption_wiseft_se_cross, Baseline_caption_wiseft_se_text, Baseline_caption_wiseft_se_text_cross
from configs import get_cfg_default
import logging


MODELS = {
    'baseline': Baseline,
    'baseline_cattn': Baseline_cattn,
    'baseline_cattn_vocabloss': Baseline_cattn_vocabloss,
    'baseline_cattn_vocabloss_wotextloss': Baseline_cattn_vocabloss_wotextloss,
    'baseline_cattn_vocabloss_cpvocab': Baseline_cattn_vocabloss_cpvocab,
    'baseline_cattn_vocabloss_shembed': Baseline_cattn_vocabloss_shembed,
    'baseline_cattn_vocabloss_shembed_mul': Baseline_cattn_vocabloss_shembed_mul,
    'baseline_cattn_vocabloss_shembed_zsinit': Baseline_cattn_vocabloss_shembed_zsinit,
    'baseline_cattn_vocabloss_shembed_zsinit_fixed': Baseline_cattn_vocabloss_shembed_zsinit_fixed,
    'baseline_cattn_vocabloss_shembed_lscale': Baseline_cattn_vocabloss_shembed_lscale,
    'baseline_cattn_vocabloss_shembed_zsinit_optimfc': Baseline_cattn_vocabloss_shembed_zsinit_optimfc,
    'baseline_sam': Baseline_sam,
    'lpclip': lpclip,
    'lpsam': lpsam,
    'baseline_cattn_embedloss': Baseline_cattn_embedloss,
    'baseline_cattn_vocabloss_shembed_zsinit_2xcattn': Baseline_cattn_vocabloss_shembed_zsinit_2xcattn,
    'baseline_cattn_vocabloss_shembed_zsinit_2xcattn_pe': Baseline_cattn_vocabloss_shembed_zsinit_2xcattn_pe,
    'baseline_cattn_vl_pd': Baseline_cattn_vl_pd,
    'baseline_cattn_vocabloss_shembed_zsinit_fixedfirst': Baseline_cattn_vocabloss_shembed_zsinit_fixedfirst,
    'baseline_cattn_vocabloss_shembed_zsinit_textaug': Baseline_cattn_vocabloss_shembed_zsinit_textaug,
    'baseline_cattn_vocabloss_shembed_zsinit_lscale': Baseline_cattn_vocabloss_shembed_zsinit_lscale,
    'baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft': Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft,
    'baseline_cattn_vocabloss_shembed_zsinit_mul_lscale_wiseft': Baseline_cattn_vocabloss_shembed_zsinit_mul_lscale_wiseft,
    'baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_nxcattn': Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_nxcattn,
    'baseline_cattn_vocabloss_shembed_zsinit_lscale_lnable_wiseft_nxcattn': Baseline_cattn_vocabloss_shembed_zsinit_lscale_lnable_wiseft_nxcattn,
    'baseline_cattn_coophead': Baseline_cattn_coophead,
    'baseline_cattn_vocabloss_shembed_zsinit_lscale_fixed': Baseline_cattn_vocabloss_shembed_zsinit_lscale_fixed,
    'baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_wotextloss': Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_wotextloss,
    'baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_wiseftval': Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft,
    'baseline_sattn': Baseline_sattn,
    'baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_add': Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft_add,
    'baseline_cattn_wiseft_template_ensemble': Baseline_cattn_wiseft_template_ensemble,
    'baseline_caption_wiseft': Baseline_caption_wiseft,
    'baseline_caption_wiseft_multi_stream': Baseline_caption_wiseft_multi_stream,
    'baseline_abla_caption_only': Baseline_abla_caption_only,
    'baseline_abla_multi_wo_caption': Baseline_abla_multi_wo_caption,
    'baseline_caption_wiseft_multi_stream_headval': Baseline_caption_wiseft_multi_stream,
    'baseline_caption_wiseft_nxcattn': Baseline_caption_wiseft_nxcattn,
    'baseline_caption_wiseft_nxcattn_add': Baseline_caption_wiseft_nxcattn_add,
    'baseline_caption_wiseft_nxcattn_auxi': Baseline_caption_wiseft_nxcattn_auxi,
    'baseline_caption_wiseft_multi_stream_projector': Baseline_caption_wiseft_multi_stream_projector,
    'baseline_caption_wiseft_multi_stream_bert': Baseline_caption_wiseft_multi_stream_bert,
    'baseline_caption_wiseft_multi_stream_t5': Baseline_caption_wiseft_multi_stream_t5,
    'baseline_caption_wiseft_multi_stream_rn': Baseline_caption_wiseft_multi_stream_rn,
    'baseline_caption_wiseft_lora': Baseline_caption_wiseft_lora,
    'baseline_caption_wiseft_lora_fixedfirst': Baseline_caption_wiseft_lora_fixedfirst,
    'baseline_caption_wiseft_se_pre_all': Baseline_caption_wiseft_se_pre_all,
    'baseline_caption_wiseft_se_post': Baseline_caption_wiseft_se_post,
    'baseline_caption_wiseft_se_cross': Baseline_caption_wiseft_se_cross,
    'baseline_caption_wiseft_se_text': Baseline_caption_wiseft_se_text,
    'baseline_caption_wiseft_se_text_cross': Baseline_caption_wiseft_se_text_cross
}


def print_args(args, cfg):
    logger = logging.getLogger(cfg.TRAINER.NAME)
    logger.info("***************")
    logger.info("** Arguments **")
    logger.info("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        logger.info("{}: {}".format(key, args.__dict__[key]))
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    # if args.trainer:
    #     cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.dist_train:
        cfg.TRAIN.DIST_TRAIN = args.dist_train


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

    cfg.INPUT.TEXT_AUG = "hand_crafted"
    cfg.INPUT.NUM_VIEWS = 1

    cfg.OPTIM.LR_FC_RATIO = 1.

    cfg.MODEL.SAM = CN()
    cfg.MODEL.SAM.CHECKPOINT = ''
    cfg.MODEL.SAM.NAME = ''

    cfg.TRAIN.FIX_EPOCH = 0

    cfg.OPTIM.MAX_ITER = 12800
    cfg.OPTIM.WARMUP_ITER = 50
    cfg.OPTIM.WARMUP_LR = 1e-5

    cfg.MODEL.BONDER = CN()
    cfg.MODEL.BONDER.DEPTH = 1
    cfg.MODEL.BONDER.NUM_Q = 32
    cfg.MODEL.CAPTION = False
    cfg.TRAIN.VAL_WISEFT = True

    cfg.MODEL.TEXT = CN()
    cfg.MODEL.TEXT.ENCODER = 'clip-text'

    cfg.MODEL.LORA = CN()
    cfg.MODEL.LORA.RANK = 1
    cfg.MODEL.LORA.ALPHA = 0.

    cfg.DATA_SEED = 1

    cfg.OPTIM.LORA_OPTIM = False    # whether use optim_lora
    cfg.OPTIM.LR_LORA = 1e-4

    cfg.SIMPLE_SEED = False  # whether trigger 3 seeds in once


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    return cfg