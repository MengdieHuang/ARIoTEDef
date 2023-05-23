

CUDA_VISIBLE_DEVICES=1 python stdtrain-retrain-maggie-20230523.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 5 --seq2seq_epochs 3 --relabel_rounds 2 > /home/huan1932/ARIoTEDef/result/log/retrain-infection-seq2seq-maggie-20230522.log 2>&1