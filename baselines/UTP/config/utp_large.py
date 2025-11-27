import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from ..arch import UTP
from ..data import BLASTDatasetMixUp
from ..runner import UTPRunner
from ..loss import fake_loss


############################## Hot Parameters ##############################
# Dataset & Metrics configuration
# Model architecture and parameters

MODEL_ARCH = UTP

context_length = 1024
predict_length = 64 # ref: chronos-bolt-base/config.json

MODEL_PARAM = {
    "embed_dim": 768,
    "num_heads": 12,
    "mlp_hidden_dim": 3072,
    "num_layers": 12,
    "use_rope_x": True,
    "rope_base": 10000.0,
    "base_context_length": 1024,
    "use_reg_token": True,
    "asinh_transform": True,
}
DATA_NAME = "BLAST"

NUM_ITERATIONS = 200_000 # 总轮数
VAL_ITERATION_INTERVAL = 5_000 # 每VAL_ITERATION_INTERVAL执行一次验证

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'UTP Base | Debug: Data'
CFG.GPU_NUM = 4 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = UTPRunner

# CFG.ENV = EasyDict() # Environment settings. Default: None
# CFG.ENV.SEED = 2025 # Random seed. Default: None

############################## Model Configuration ################################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.DTYPE= 'bfloat16'

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({})

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.COMPILE_MODEL = False
CFG.TRAIN.NUM_ITERATIONS = NUM_ITERATIONS
CFG.TRAIN.CKPT_SAVE_STRATEGY = VAL_ITERATION_INTERVAL * 1 # 保存策略，每VAL_ITERATION_INTERVAL * 1 保存一次模型
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_ITERATIONS)])
)
CFG.TRAIN.LOSS = fake_loss
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 1e-4,
    "betas": (0.9, 0.95),
    "fused": True
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineWarmup"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'num_warmup_steps': int(NUM_ITERATIONS / 100 * 10), # 10%的warmup启动比例
    'num_training_steps': NUM_ITERATIONS,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 1.0
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.DATA.PREFETCH = True
CFG.TRAIN.GRAD_ACCUMULATION_STEPS = 1

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = VAL_ITERATION_INTERVAL
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()
# Evaluation parameters
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = BLASTDatasetMixUp
CFG.DATASET.PARAM = EasyDict({
    'context_length': context_length,
    'target_length': predict_length,
    'num_valid_samples': 1000
})

############################## Inference Configuration ##############################
CFG.INFERENCE = EasyDict()
CFG.INFERENCE.GENERATION_PARAMS = EasyDict({
})
