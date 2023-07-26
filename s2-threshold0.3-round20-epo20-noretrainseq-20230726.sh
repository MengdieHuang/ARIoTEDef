
# without retrain seq2seq
CUDA_VISIBLE_DEVICES=1 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 1 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.5  > /home/huan1932/ARIoTEDef/log/20230726/stdtrainps-stdtrainseq-strategy2-round20-seed1-seq2seq_threshold-0.5.log 2>&1

CUDA_VISIBLE_DEVICES=1 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 0 --ps_epochs 30 --batchsize 32 --seq2seq_epochs 30 --relabel_rounds 40 --seq2seq_threshold 0.5  > /home/huan1932/ARIoTEDef/log/20230726/stdtrainps-stdtrainseq-strategy2-epo30-round40-seed0-seq2seq_threshold-0.5.log 2>&1



CUDA_VISIBLE_DEVICES=2 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 0 --ps_epochs 5 --batchsize 32 --seq2seq_epochs 5 --relabel_rounds 20 --seq2seq_threshold 0.5  > /home/huan1932/ARIoTEDef/log/20230726/test.log 2>&1

# --rec_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/attack-detector-acc-0.9542.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/seed-1/20230612/advs1/00003/infection-seq2seq-acc-0.9309.h5