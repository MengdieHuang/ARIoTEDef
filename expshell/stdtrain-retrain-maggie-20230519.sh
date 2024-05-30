
# CUDA_VISIBLE_DEVICES=1 python stdtrain-retrain-maggie-20230519.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 50 > /home/huan1932/ARIoTEDef/result/log/stdtrain-retrain-maggie-20230519.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python stdtrain-retrain-maggie-20230519.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 50 > /home/huan1932/ARIoTEDef/result/log/advattack-retrain-maggie-20230519.log 2>&1


CUDA_VISIBLE_DEVICES=1 python stdtrain-retrain-maggie-20230519.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 50 --seq2seq_epochs 50 > /home/huan1932/ARIoTEDef/result/log/train-infection-seq2seq-maggie-20230519.log 2>&1