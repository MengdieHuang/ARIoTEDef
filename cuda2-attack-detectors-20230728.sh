


#---------------------------------------

CUDA_VISIBLE_DEVICES=1 python taskluncher-20230728.py --stdtrain_seq2seq --attack boundary --eps 9.0 --eps_step 9.0 --max_iter 5000 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 10 --seq2seq_threshold 0.3  --rec_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/action-detector-acc-0.9773.h5 > /home/huan1932/ARIoTEDef/log/20230728/stdtrain-seq-boundary-eps9.0-epo40-iter5000-round10-embedrv1.log 2>&1


CUDA_VISIBLE_DEVICES=1 python taskluncher-20230728.py --stdtrain_seq2seq --attack boundary --eps 6.0 --eps_step 6.0 --max_iter 5000 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 10 --seq2seq_threshold 0.3  --rec_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/action-detector-acc-0.9773.h5 > /home/huan1932/ARIoTEDef/log/20230728/stdtrain-seq-boundary-eps6.0-epo40-iter5000-round10-embedrv1.log 2>&1



