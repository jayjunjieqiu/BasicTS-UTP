import os
import sys
import torch
sys.path.append(os.path.abspath(__file__ + '/../../../..'))
from baselines.UTP2.arch.utp import UTP2, UTP2Config

def convert_ckpt(ckpt_path: str, model_param: dict, save_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state_dict = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[len('module.') :]
        if nk.startswith('utp2.'):
            nk = nk[len('utp2.') :]
        elif nk.startswith('utp.'):
            nk = nk[len('utp.') :]
        new_state_dict[nk] = v
    
    config = UTP2Config(**model_param)
    model = UTP2(config)
    model.load_state_dict(new_state_dict, strict=True)
    UTP2.save_model(model, save_path)
    return save_path

# Example usage
ckpt_path = '/data/junjieqiu/BasicTS-0.5.8/checkpoints/UTP2Pretrain/BLAST_100000/387a1cc0d30e8971b693fbf8e7f0ea95/UTP2Pretrain_100000.pt'
save_path = '/data/junjieqiu/BasicTS-0.5.8/checkpoints/UTP2Pretrain/BLAST_100000/387a1cc0d30e8971b693fbf8e7f0ea95/factost_utp2_mini_100000.pt'
# model_param = {
#     "quantiles": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
#     "patch_size": 16,
#     "rope_percentage": 0.75,
#     "max_input_patches": 128,
#     "max_output_patches": 16,
#     "hidden_size": 256,
#     "intermediate_size": 1024,
#     "num_layers": 3,
#     "num_attention_heads": 4,
#     "use_arcsinh": True,
#     "rope_theta": 10000.0,
#     "dropout": 0.1,
# }
model_param = {
    "quantiles": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
    "patch_size": 16,
    "rope_percentage": 0.75,
    "max_input_patches": 128,
    "max_output_patches": 16,
    "hidden_size": 384,
    "intermediate_size": 1536,
    "num_layers": 4,
    "num_attention_heads": 6,
    "use_arcsinh": True,
    "rope_theta": 10000.0,
    "dropout": 0.1,
}
# model_param = {
#     "quantiles": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
#     "patch_size": 16,
#     "rope_percentage": 0.75,
#     "max_input_patches": 128,
#     "max_output_patches": 16,
#     "hidden_size": 512,
#     "intermediate_size": 2048,
#     "num_layers": 6,
#     "num_attention_heads": 8,
#     "use_arcsinh": True,
#     "rope_theta": 10000.0,
#     "dropout": 0.1,
# }

if __name__ == "__main__":
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    converted_path = convert_ckpt(ckpt_path, model_param, save_path)
    print('Converted ckpt saved to:', converted_path)
    
    # Verify by loading the converted ckpt
    model2 = UTP2.load_model(converted_path, map_location='cpu')
    context = torch.randn(1, 32 * 16) # batch_size, input_length
    total_params = sum(p.numel() for p in model2.parameters())
    print(f"Number of parameters: {total_params:,}")
    # Need context_mask for UTP2 forward/predict usually
    # But predict method handles list of tensors. If tensor, it expects (B, L)
    # predict method signature: predict(context, prediction_length)
    pred = model2.predict(context, prediction_length=8 * 16)
    print("Dummy prediction passed successfully.")
