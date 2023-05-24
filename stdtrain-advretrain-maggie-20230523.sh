CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230523.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 50 --seq2seq_epochs 30 --relabel_rounds 10 --retrainset_mode olds1 > /home/huan1932/ARIoTEDef/result/log/olds1-advretrain-infection-seq2seq-maggie-20230523.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230523.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 2 --seq2seq_epochs 2 --relabel_rounds 6 --retrainset_mode olds1 > /home/huan1932/ARIoTEDef/result/log/olds1-advretrain-infection-seq2seq-maggie-20230523.log 2>&1


# CUDA_VISIBLE_DEVICES=2 python stdtrain-advretrain-maggie-20230523.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 50 --seq2seq_epochs 30 --relabel_rounds 10 --retrainset_mode advs1 > /home/huan1932/ARIoTEDef/result/log/advs1-advretrain-infection-seq2seq-maggie-20230523.log 2>&1