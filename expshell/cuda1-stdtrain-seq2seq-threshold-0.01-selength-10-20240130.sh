


# # train per-step detectors
# CUDA_VISIBLE_DEVICES=0 python taskluncher-20240130.py --stdtrain_pedetector --stdtrain_seq2seq --sequence_length 10 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --seq2seq_threshold 0.01 > /home/huan1932/ARIoTEDef-Actual/log/20240130/stdtrain-seq2seq-seq2seqthreshold-0.01-seqlength-10-seed-0-epo40.log 2>&1


# CUDA_VISIBLE_DEVICES=1 python taskluncher-20240130.py --stdtrain_pedetector --stdtrain_seq2seq --sequence_length 10 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 1 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --seq2seq_threshold 0.01 > /home/huan1932/ARIoTEDef-Actual/log/20240130/stdtrain-seq2seq-seq2seqthreshold-0.01-seqlength-10-seed-1-epo40.log 2>&1


# CUDA_VISIBLE_DEVICES=2 python taskluncher-20240130.py --stdtrain_pedetector --stdtrain_seq2seq --sequence_length 10 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 2 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --seq2seq_threshold 0.01 > /home/huan1932/ARIoTEDef-Actual/log/20240130/stdtrain-seq2seq-seq2seqthreshold-0.01-seqlength-10-seed-2-epo40.log 2>&1











# #  round 20 epoch 40
# CUDA_VISIBLE_DEVICES=0 python taskluncher-20240130.py --advset_mode advset1 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240130/retrain-adv-epoch-40-seq2seqthreshold-0.01-round-20-seed-0-strategy2.log 2>&1


# CUDA_VISIBLE_DEVICES=0 python taskluncher-20240130.py --advset_mode advset1 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 1 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240130/retrain-adv-epoch-40-seq2seqthreshold-0.01-round-20-seed-1-strategy2.log 2>&1


# CUDA_VISIBLE_DEVICES=0 python taskluncher-20240130.py --advset_mode advset1 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 2 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240130/retrain-adv-epoch-40-seq2seqthreshold-0.01-round-20-seed-2-strategy2.log 2>&1

# #  round 20 epoch 30
# CUDA_VISIBLE_DEVICES=0 python taskluncher-20240130.py --advset_mode advset1 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 0 --ps_epochs 30 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240130/retrain-adv-epoch-30-seq2seqthreshold-0.01-round-20-seed-0-strategy2.log 2>&1


# CUDA_VISIBLE_DEVICES=0 python taskluncher-20240130.py --advset_mode advset1 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 1 --ps_epochs 30 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240130/retrain-adv-epoch-30-seq2seqthreshold-0.01-round-20-seed-1-strategy2.log 2>&1


# CUDA_VISIBLE_DEVICES=2 python taskluncher-20240130.py --advset_mode advset1 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 2 --ps_epochs 30 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240130/retrain-adv-epoch-30-seq2seqthreshold-0.01-round-20-seed-2-strategy2.log 2>&1


#round 20 epoch 20
CUDA_VISIBLE_DEVICES=0 python taskluncher-20240130.py --advset_mode advset1 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 0 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240130/retrain-adv-epoch-20-seq2seqthreshold-0.01-round-20-seed-0-strategy2.log 2>&1


CUDA_VISIBLE_DEVICES=0 python taskluncher-20240130.py --advset_mode advset1 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 1 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240130/retrain-adv-epoch-20-seq2seqthreshold-0.01-round-20-seed-1-strategy2.log 2>&1


CUDA_VISIBLE_DEVICES=2 python taskluncher-20240130.py --advset_mode advset1 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 2 --ps_epochs 20 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240130/retrain-adv-epoch-20-seq2seqthreshold-0.01-round-20-seed-2-strategy2.log 2>&1