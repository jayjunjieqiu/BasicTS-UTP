import os
import sys
from easydict import EasyDict

from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils.serialization import get_regular_settings
from basicts.metrics import masked_mae, masked_rmse, masked_mape
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from ...arch.model import UTP2Pretrain, UTP2Config
from ...runner.runner import UTP2Runner


############################## Hot Parameters ##############################
# Dataset & Metrics configuration
# Model architecture and parameters

MODEL_ARCH = UTP2Pretrain

CONTEXT_LENGTH = None
PREDICTION_LENGTH = None

UTP2_CONFIG = UTP2Config(
    quantiles=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
    patch_size=16,
    rope_percentage=0.75,
    max_input_patches=128,
    max_output_patches=16,
    hidden_size=384,
    intermediate_size=1536,
    num_layers=4,
    num_attention_heads=6,
    rope_theta=10000.0,
    attention_dropout=0.0,
)

MODEL_PARAM = {
    "config": UTP2_CONFIG
}
DATA_NAME = "METR-LA"

NUM_ITERATIONS = None # 总轮数

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'UTP2 Base'
CFG.GPU_NUM = 8 # Number of GPUs to use (0 for CPU mode)
# CFG.GPU_NUM = 8 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = UTP2Runner

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
regular_settings['TRAIN_VAL_TEST_RATIO'] = [0.6, 0.2, 0.2]

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
    'overlap': False
})
CFG.TEST = EasyDict()
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = True
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
                                'RMSE': masked_rmse,
                                'MAPE': masked_mape
                            })
CFG.METRICS.NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data

############################## Inference Configuration ##############################
CFG.INFERENCE = EasyDict()
CFG.INFERENCE.GENERATION_PARAMS = EasyDict({
})
