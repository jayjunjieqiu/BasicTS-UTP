import os
import sys
import torch
sys.path.append(os.path.abspath(__file__ + '/../../../..'))
from baselines.UTP.arch.utp import UTPModel, UTPModelConfig

def convert_ckpt(ckpt_path: str, model_param: dict, save_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state_dict = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.') :]
        if nk.startswith('utp.'):
            nk = nk[len('utp.') :]
        new_state_dict[nk] = v
    config = UTPModelConfig(**model_param)
    model = UTPModel(config)
    model.load_state_dict(new_state_dict, strict=True)
    UTPModel.save_model(model, save_path)
    return save_path

# ckpt_path = '/data/junjieqiu/BasicTS-0.5.8/checkpoints/UTP/BLAST_100000/utp_exp2_base/UTP_best_val_loss.pt'
# save_path = '/data/junjieqiu/BasicTS-0.5.8/checkpoints/UTP/BLAST_100000/utp_exp2_base/converted_best_val_loss.pt'
ckpt_path = '/data/junjieqiu/BasicTS-0.5.8/checkpoints/UTP/BLAST_100000/d83b96a5011f904ae0616f6feb200d63/UTP_best_val_loss.pt'
save_path = '/data/junjieqiu/BasicTS-0.5.8/checkpoints/UTP/BLAST_100000/d83b96a5011f904ae0616f6feb200d63/converted_best_val_loss.pt'
# model_param = {
#     "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     "hidden_size": 384,
#     "intermediate_size": 1536,
#     "num_layers": 6,
#     "rope_percentage": 0.75,
#     "num_attention_heads": 12,
#     "rope_theta": 10000.0,
#     "attention_dropout": 0.0,
# }
model_param = {
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "hidden_size": 192,
    "intermediate_size": 768,
    "num_layers": 4,
    "rope_percentage": 0.75,
    "num_attention_heads": 6,
    "rope_theta": 10000.0,
    "attention_dropout": 0.0,
}

os.makedirs(os.path.dirname(save_path), exist_ok=True)
converted_path = convert_ckpt(ckpt_path, model_param, save_path)
print('Converted ckpt saved to:', converted_path)

# Verify by loading the converted ckpt
model2 = UTPModel.load_model(converted_path, map_location='cpu')
context = torch.randn(1, 1, 1024)
pred = model2.predict(context, prediction_length=64)
print("Dummy prediction passed successfully.")
