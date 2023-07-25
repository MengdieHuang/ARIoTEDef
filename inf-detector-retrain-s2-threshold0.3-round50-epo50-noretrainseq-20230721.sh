
# without retrain seq2seq
CUDA_VISIBLE_DEVICES=0 python inf-detector-retrainseq2seq-20230721.py --strategy strategy2 --patience 50 --timesteps 1 --seed 0 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 50 --relabel_rounds 50 --seq2seq_threshold 0.3 --rec_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/attack-detector-acc-0.9542.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/seed-1/20230612/advs1/00003/infection-seq2seq-acc-0.9309.h5 > /home/huan1932/ARIoTEDef/log/20230721/noretrainseq2seq-strategy2-round50-seed0-seq2seq_threshold-0.3-epo50.log 2>&1


CUDA_VISIBLE_DEVICES=0 python inf-detector-retrainseq2seq-20230721.py --strategy strategy2 --patience 50 --timesteps 1 --seed 1 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 50 --relabel_rounds 50 --seq2seq_threshold 0.3 --rec_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/attack-detector-acc-0.9542.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/seed-1/20230612/advs1/00003/infection-seq2seq-acc-0.9309.h5 > /home/huan1932/ARIoTEDef/log/20230721/noretrainseq2seq-strategy2-round50-seed1-seq2seq_threshold-0.3-epo50.log 2>&1


CUDA_VISIBLE_DEVICES=0 python inf-detector-retrainseq2seq-20230721.py --strategy strategy2 --patience 50 --timesteps 1 --seed 0 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 50 --relabel_rounds 100 --seq2seq_threshold 0.3 --rec_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/attack-detector-acc-0.9542.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/seed-1/20230612/advs1/00003/infection-seq2seq-acc-0.9309.h5 > /home/huan1932/ARIoTEDef/log/20230721/noretrainseq2seq-strategy2-round100-seed0-seq2seq_threshold-0.3-epo50.log 2>&1


CUDA_VISIBLE_DEVICES=0 python inf-detector-retrainseq2seq-20230721.py --strategy strategy2 --patience 50 --timesteps 1 --seed 1 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 50 --relabel_rounds 100 --seq2seq_threshold 0.3 --rec_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/attack-detector-acc-0.9542.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/seed-1/20230612/advs1/00003/infection-seq2seq-acc-0.9309.h5 > /home/huan1932/ARIoTEDef/log/20230721/noretrainseq2seq-strategy2-round100-seed1-seq2seq_threshold-0.3-epo50.log 2>&1