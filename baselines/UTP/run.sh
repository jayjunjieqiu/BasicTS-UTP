python experiments/evaluate.py -cfg baselines/UTP/evaluate_config/ETTh1/utp_base.py -g 0 -ckpt checkpoints/UTP/BLAST_200000/faeacf6e79ed12b50e1b9beb3bdf7195/UTP_best_val_loss.pt -ctx 1024 -pred 96

python experiments/evaluate.py -cfg baselines/UTP/evaluate_config/ETTh1/utp_large.py -g 6 -ckpt checkpoints/UTP/BLAST_200000/5c43ef4776fe8ba5b84bdeb5401ce179/UTP_best_val_loss.pt -ctx 1024 -pred 96
