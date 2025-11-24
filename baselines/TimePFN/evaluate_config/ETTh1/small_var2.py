import os
import sys
from easydict import EasyDict

from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset
from basicts.scaler.z_score_scaler import ZScoreScaler
from basicts.utils.serialization import get_regular_settings
from basicts.metrics import masked_mae, masked_mse
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from ...arch.model import TimePFN
from ...runner.runner import TimePFNRunner

MODEL_ARCH = TimePFN

CONTEXT_LENGTH = None
PREDICTION_LENGTH = None

DATA_NAME = "ETTh1"

CFG = EasyDict()
CFG.DESCRIPTION = 'TimePFN Base | Debug: Data'
CFG.GPU_NUM = 8
CFG.RUNNER = TimePFNRunner

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = {
    "embed_dim": 384,
    "pe_dim": 192,
    "num_heads": 12,
    "mlp_hidden_dim": 768,
    "num_layers": 6,
    "use_rope_x": True,
    "rope_base": 10000.0,
    "use_y_attn": False,
}
CFG.MODEL.DTYPE= 'float32'

CFG.TRAIN = EasyDict()
CFG.TRAIN.COMPILE_MODEL = False
CFG.TRAIN.NUM_ITERATIONS = None
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_ITERATIONS)])
)

regular_settings = get_regular_settings(dataset_name=DATA_NAME)

CFG.DATASET = EasyDict()
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

CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': regular_settings['TRAIN_VAL_TEST_RATIO'][0],
    'norm_each_channel': regular_settings['NORM_EACH_CHANNEL'],
    'rescale': regular_settings['RESCALE'],
})

CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'MSE': masked_mse,
})
CFG.METRICS.NULL_VAL = regular_settings['NULL_VAL']

CFG.INFERENCE = EasyDict()
CFG.INFERENCE.GENERATION_PARAMS = EasyDict({})
