import os
import sys
from easydict import EasyDict

from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils.serialization import get_regular_settings
from basicts.metrics import masked_mae, masked_mse
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from ...arch.model import UTP
from ...runner.runner import UTPRunner


############################## Hot Parameters ##############################
# Dataset & Metrics configuration
# Model architecture and parameters

MODEL_ARCH = UTP

CONTEXT_LENGTH = None
PREDICTION_LENGTH = None

MODEL_PARAM = {
    "embed_dim": 384,
    "num_heads": 12,
    "mlp_hidden_dim": 1536,
    "num_layers": 12,
    "use_rope_x": True,
    "rope_base": 10000.0,
    "base_context_length": 1024,
    "use_reg_token": True,
    "asinh_transform": True,
    "quantiles": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
}
DATA_NAME = "Weather"

NUM_ITERATIONS = None # 总轮数

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'UTP Tiny | Debug: Data'
CFG.GPU_NUM = 8 # Number of GPUs to use (0 for CPU mode)
# CFG.GPU_NUM = 8 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = UTPRunner

############################## Model Configuration ################################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.DTYPE= 'float32'

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.COMPILE_MODEL = False
CFG.TRAIN.NUM_ITERATIONS = NUM_ITERATIONS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_ITERATIONS)])
)

regular_settings = get_regular_settings(dataset_name=DATA_NAME)

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': regular_settings['TRAIN_VAL_TEST_RATIO'],
    'input_len': CONTEXT_LENGTH,
    'output_len': PREDICTION_LENGTH,
    'overlap': True
})
CFG.TEST = EasyDict()
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.SHUFFLE = False

############################## TEST Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': regular_settings['TRAIN_VAL_TEST_RATIO'][0],
    'norm_each_channel': regular_settings['NORM_EACH_CHANNEL'],
    'rescale': regular_settings['RESCALE'],
})

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MSE': masked_mse,
                            })
CFG.METRICS.NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data

############################## Inference Configuration ##############################
CFG.INFERENCE = EasyDict()
CFG.INFERENCE.GENERATION_PARAMS = EasyDict({
})
