CUDA_VISIBLE_DEVICES=0 python stdtrain-advretrain-maggie-20230615.py --strategy strategy1 --patience 50 --timesteps 1 --seed 1 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 50 --relabel_rounds 50 --retrainset_mode advs1 --rec_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/attack-detector-acc-0.9542.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230609/advs1/00008/infection-seq2seq-acc-0.9233.h5 > /home/huan1932/ARIoTEDef/log/20230615/strategy1-retrain-round50-infection-seq2seq-seed1-maggie-20230615.log 2>&1

CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230615.py --strategy strategy2 --patience 50 --timesteps 1 --seed 1 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 50 --relabel_rounds 50 --retrainset_mode advs1 --rec_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/attack-detector-acc-0.9542.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230609/advs1/00008/infection-seq2seq-acc-0.9233.h5 > /home/huan1932/ARIoTEDef/log/20230615/strategy2-retrain-round50-infection-seq2seq-seed1-maggie-20230615.log 2>&1

CUDA_VISIBLE_DEVICES=2 python stdtrain-advretrain-maggie-20230615.py --strategy strategy3 --patience 50 --timesteps 1 --seed 1 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 50 --relabel_rounds 50 --retrainset_mode advs1 --rec_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/attack-detector-acc-0.9542.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230609/advs1/00008/infection-seq2seq-acc-0.9233.h5 > /home/huan1932/ARIoTEDef/log/20230615/strategy3-retrain-round50-infection-seq2seq-seed1-maggie-20230615.log 2>&1

# strategy4-groundtruth

# CUDA_VISIBLE_DEVICES=2 python stdtrain-advretrain-maggie-20230615.py --stdtrain_seq2seq --patience 50 --timesteps 1 --seed 1 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 50 --relabel_rounds 50 --retrainset_mode advs1 --rec_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/attack-detector-acc-0.9542.h5 > /home/huan1932/ARIoTEDef/log/20230615/stdtrain-seq2seq-epo50-seed1-maggie-20230615.log 2>&1

CUDA_VISIBLE_DEVICES=2 python stdtrain-advretrain-maggie-20230615.py --strategy strategy4-groundtruth --patience 50 --timesteps 1 --seed 1 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 50 --relabel_rounds 50 --retrainset_mode advs1 --rec_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230606/advs1/00002/attack-detector-acc-0.9542.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/seed-0/20230609/advs1/00008/infection-seq2seq-acc-0.9233.h5 > /home/huan1932/ARIoTEDef/log/20230615/strategy4-groundtruth-retrain-round50-infection-seq2seq-seed1-maggie-20230615.log 2>&1


# /home/huan1932/ARIoTEDef/result/seed-1/20230615/advs1/00003/infection-seq2seq-acc-0.9309.h5