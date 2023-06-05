CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230604.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 30 --seq2seq_epochs 30 --relabel_rounds 50 --retrainset_mode advs1 > /home/huan1932/ARIoTEDef/result/log/advs1-advretrain-round50-infection-seq2seq-maggie-20230604.log 2>&1


CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230605.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 2 --seq2seq_epochs 2 --relabel_rounds 5 --retrainset_mode advs1