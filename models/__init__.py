from .baseline import Baseline
from .lpclip import lpclip
from .baseline_ic import Baseline_ic
from .baseline_cattn import Baseline_cattn
# from .baseline_cattn_vocabloss import Baseline_cattn_vocabloss, Baseline_cattn_vocabloss_wotextloss, \
#     Baseline_cattn_vocabloss_cpvocab, Baseline_cattn_vocabloss_shembed,Baseline_cattn_vocabloss_shembed_mul, \
#     Baseline_cattn_vocabloss_shembed_zsinit, Baseline_cattn_vocabloss_shembed_zsinit_fixed, \
#     Baseline_cattn_vocabloss_shembed_lscale, Baseline_cattn_vocabloss_shembed_zsinit_optimfc, \
#     Baseline_cattn_vocabloss_shembed_zsinit_2xcattn, Baseline_cattn_vocabloss_shembed_zsinit_2xcattn_pe, \
#     Baseline_cattn_vocabloss_shembed_zsinit_fixedfirst, Baseline_cattn_vocabloss_shembed_zsinit_textaug, \
#     Baseline_cattn_vocabloss_shembed_zsinit_lscale, Baseline_cattn_vocabloss_shembed_zsinit_lscale_wiseft, \
#     Baseline_cattn_vocabloss_shembed_zsinit_mul_lscale_wiseft
from .baseline_cattn_vocabloss import *
from .baseline_sam import Baseline_sam
from .lpsam import lpsam
from .baseline_cattn_embedloss import Baseline_cattn_embedloss
from .baseline_cattn_vl_pd import Baseline_cattn_vl_pd
from .baseline_cattn_coophead import Baseline_cattn_coophead