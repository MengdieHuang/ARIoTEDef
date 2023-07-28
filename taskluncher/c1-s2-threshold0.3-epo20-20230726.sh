# --rec_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/attack-detector-acc-0.9542.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/seed-1/20230612/advs1/00003/infection-seq2seq-acc-0.9309.h5

# without retrain seq2seq
CUDA_VISIBLE_DEVICES=1 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 0 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 20 --relabel_rounds 20 --seq2seq_threshold 0.3  > /home/huan1932/ARIoTEDef/log/20230726/stdtrainps-stdtrainseq-strategy2-epo20-round20-seed0-seq2seqthreshold-0.3.log 2>&1

CUDA_VISIBLE_DEVICES=1 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 0 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 20 --relabel_rounds 40 --seq2seq_threshold 0.3  > /home/huan1932/ARIoTEDef/log/20230726/stdtrainps-stdtrainseq-strategy2-epo20-round40-seed0-seq2seqthreshold-0.3.log 2>&1

CUDA_VISIBLE_DEVICES=1 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 0 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 20 --relabel_rounds 60 --seq2seq_threshold 0.3  > /home/huan1932/ARIoTEDef/log/20230726/stdtrainps-stdtrainseq-strategy2-epo20-round60-seed0-seq2seqthreshold-0.3.log 2>&1


#----------------------------
CUDA_VISIBLE_DEVICES=1 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 1 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 20 --relabel_rounds 20 --seq2seq_threshold 0.3  > /home/huan1932/ARIoTEDef/log/20230726/stdtrainps-stdtrainseq-strategy2-epo20-round20-seed1-seq2seqthreshold-0.3.log 2>&1

CUDA_VISIBLE_DEVICES=1 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 1 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 20 --relabel_rounds 40 --seq2seq_threshold 0.3  > /home/huan1932/ARIoTEDef/log/20230726/stdtrainps-stdtrainseq-strategy2-epo20-round40-seed1-seq2seqthreshold-0.3.log 2>&1

CUDA_VISIBLE_DEVICES=1 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 1 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 20 --relabel_rounds 60 --seq2seq_threshold 0.3  > /home/huan1932/ARIoTEDef/log/20230726/stdtrainps-stdtrainseq-strategy2-epo20-round60-seed1-seq2seqthreshold-0.3.log 2>&1


#----------------------------
CUDA_VISIBLE_DEVICES=1 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 2 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 20 --relabel_rounds 20 --seq2seq_threshold 0.3  > /home/huan1932/ARIoTEDef/log/20230726/stdtrainps-stdtrainseq-strategy2-epo20-round20-seed2-seq2seqthreshold-0.3.log 2>&1

CUDA_VISIBLE_DEVICES=1 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 2 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 20 --relabel_rounds 40 --seq2seq_threshold 0.3  > /home/huan1932/ARIoTEDef/log/20230726/stdtrainps-stdtrainseq-strategy2-epo20-round40-seed2-seq2seqthreshold-0.3.log 2>&1

CUDA_VISIBLE_DEVICES=1 python taskluncher-20230725.py --stdtrain_pedetector --stdtrain_seq2seq --strategy strategy2 --patience 50 --timesteps 1 --seed 2 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 20 --relabel_rounds 60 --seq2seq_threshold 0.3  > /home/huan1932/ARIoTEDef/log/20230726/stdtrainps-stdtrainseq-strategy2-epo20-round60-seed2-seq2seqthreshold-0.3.log 2>&1




