
CUDA_VISIBLE_DEVICES=2 python stdtrain-advretrain-maggie-20230527.py --patience 10 --timesteps 1 --seed 1 --ps_epochs 50 --seq2seq_epochs 30 --relabel_rounds 20 --retrainset_mode advs1 > /home/huan1932/ARIoTEDef/result/log/advs1-advretrain-round50-infection-seq2seq-maggie-20230527.log 2>&1









CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230531.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 20 --seq2seq_epochs 30 --relabel_rounds 20 --retrainset_mode advs1