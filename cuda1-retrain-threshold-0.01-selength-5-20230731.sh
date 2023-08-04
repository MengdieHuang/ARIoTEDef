CUDA_VISIBLE_DEVICES=1 python taskluncher-20230730-advset2.py --sequence_length 5 --advset_mode advset2 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 5 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef/log/20230731/advset2-retraintestsetmode-adv-psepoch-40-seq2seqthreshold-0.01-seqlength-10-round-5-seed-0.log 2>&1; 

CUDA_VISIBLE_DEVICES=1 python taskluncher-20230730-advset2.py --sequence_length 5 --advset_mode advset2 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 10 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef/log/20230731/advset2-retraintestsetmode-adv-psepoch-40-seq2seqthreshold-0.01-seqlength-10-round-10-seed-0.log 2>&1

CUDA_VISIBLE_DEVICES=1 python taskluncher-20230730-advset2.py --sequence_length 5 --advset_mode advset2 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef/log/20230731/advset2-retraintestsetmode-adv-psepoch-40-seq2seqthreshold-0.01-seqlength-10-round-20-seed-0.log 2>&1







