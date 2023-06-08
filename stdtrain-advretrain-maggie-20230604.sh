# CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230604.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 30 --seq2seq_epochs 30 --relabel_rounds 50 --retrainset_mode advs1 > /home/huan1932/ARIoTEDef/result/log/advs1-advretrain-round50-infection-seq2seq-maggie-20230604.log 2>&1


# CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230605.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 30 --seq2seq_epochs 30 --relabel_rounds 50 --retrainset_mode advs1 > /home/huan1932/ARIoTEDef/result/log/advs1-advretrain-round50-infection-seq2seq-maggie-20230605.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230606.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 30 --relabel_rounds 50 --retrainset_mode advs1 > /home/huan1932/ARIoTEDef/result/log/advs1-advretrain-round50-infection-seq2seq-maggie-20230606.log 2>&1



# CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230606.py --stdtrain_pedetector --patience 50 --timesteps 1 --seed 0 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 30 --relabel_rounds 50 --retrainset_mode advs1 > /home/huan1932/ARIoTEDef/result/log/20230606/stdtrain-round50-save-perstep-detectors-maggie-20230606.log 2>&1


# CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230606.py --patience 10 --timesteps 1 --seed 0 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 30 --relabel_rounds 50 --retrainset_mode advs1 --rec_model_path /home/huan1932/ARIoTEDef/result/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/20230606/advs1/00002/attack-detector-acc-0.9542.h5 > /home/huan1932/ARIoTEDef/result/log/20230606/advs1-advtrain-round50-infection-seq2seq-maggie-20230606.log 2>&1


CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230608.py --stdtrain_seq2seq --patience 50 --timesteps 1 --seed 0 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 50 --relabel_rounds 50 --retrainset_mode advs1 --rec_model_path /home/huan1932/ARIoTEDef/result/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/20230606/advs1/00002/attack-detector-acc-0.9542.h5 > /home/huan1932/ARIoTEDef/result/log/20230608/stdtrain-infection-seq2seq-maggie-20230608.log 2>&1


CUDA_VISIBLE_DEVICES=1 python stdtrain-advretrain-maggie-20230608.py --patience 50 --timesteps 1 --seed 0 --ps_epochs 50 --batchsize 32 --seq2seq_epochs 50 --relabel_rounds 50 --retrainset_mode advs1 --rec_model_path /home/huan1932/ARIoTEDef/result/20230606/advs1/00002/reconnaissance-detector-acc-0.9655.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/20230606/advs1/00002/infection-detector-acc-0.9438.h5 --att_model_path /home/huan1932/ARIoTEDef/result/20230606/advs1/00002/attack-detector-acc-0.9542.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/20230608/advs1/00000/infection-seq2seq-acc-0.9242.h5 > /home/huan1932/ARIoTEDef/result/log/20230608/advs1-advtrain-round50-infection-seq2seq-maggie-20230608.log 2>&1