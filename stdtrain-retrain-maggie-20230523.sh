

# CUDA_VISIBLE_DEVICES=1 python stdtrain-retrain-maggie-20230523.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 50 --seq2seq_epochs 30 --relabel_rounds 20 > /home/huan1932/ARIoTEDef/result/log/retrain-infection-seq2seq-maggie-20230523.log 2>&1


CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230523.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 50 --seq2seq_epochs 30 --relabel_rounds 20 > /home/huan1932/ARIoTEDef/result/log/retrain-infection-seq2seq-maggie-20230523.log 2>&1