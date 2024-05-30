#  round 20 epoch 40 # only input adv-malicious to retrain
# CUDA_VISIBLE_DEVICES=0 python taskluncher-20240201.py --advset_mode advset1 --retrain_testset_mode adv --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --strategy strategy2 --use_prob_embedding --roundvalue_d 1 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 32 --seq2seq_epochs 40 --relabel_rounds 20 --seq2seq_threshold 0.01  --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/retrain-adv-epoch-40-seq2seqthreshold-0.01-round-20-seed-0-strategy2.log 2>&1



# CUDA_VISIBLE_DEVICES=1 python taskluncher-20240201.py --advtrain_detector --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 64 --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-epoch-40-batch-64-seed-0.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python taskluncher-20240201.py --advtrain_detector --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 128 --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-epoch-40-batch-128-seed-0.log 2>&1



# CUDA_VISIBLE_DEVICES=1 python taskluncher-20240201.py --advtrain_detector --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --patience 50 --timesteps 1 --seed 0 --ps_epochs 5 --batchsize 256 --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-epoch-5-batch-256-seed-0.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python taskluncher-20240201.py --advtrain_detector --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --patience 50 --timesteps 1 --seed 0 --ps_epochs 2 --batchsize 256 --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-epoch-2-batch-256-seed-0.log 2>&1




# CUDA_VISIBLE_DEVICES=2 python taskluncher-20240201.py --advtrain_detector --onlyadvmail --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 32 --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-epoch-40-batch-32-onlyadvmail-seed-0.log 2>&1


# CUDA_VISIBLE_DEVICES=1 python taskluncher-20240201.py --advtrain_detector --onlyadvmail --advtrain_adv_cle --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 10 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 512 --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-maxiter-10-epoch-40-batch-512-onlyadvmail-advtrain_adv_cle-seed-0.log 2>&1



# CUDA_VISIBLE_DEVICES=2 python taskluncher-20240201.py --advtrain_detector --onlyadvmail --advtrain_adv_cle --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 10 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 1024 --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-maxiter-10-epoch-40-batch-1024-onlyadvmail-advtrain_adv_cle-seed-0.log 2>&1





CUDA_VISIBLE_DEVICES=1 python taskluncher-20240201.py --advtrain_detector --onlyadvmail --advtrain_adv_cle --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 10 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 2048 --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-maxiter-10-epoch-40-batch-2048-onlyadvmail-advtrain_adv_cle-seed-0.log 2>&1




# CUDA_VISIBLE_DEVICES=2 python taskluncher-20240201.py --advtrain_detector --advtrain_adv_cle --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 10 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 2048 --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-maxiter-10-epoch-40-batch-2048-advmailbeni-advtrain_adv_cle-seed-0.log 2>&1



# CUDA_VISIBLE_DEVICES=2 python taskluncher-20240201.py --lr 0.0005 --advtrain_detector --onlyadvmail --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 10 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 4096 --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-maxiter-10-epoch-40-batch-4096-lr0.0005-onlyadvmail-advtrain_adv-seed-0-test.log 2>&1




# CUDA_VISIBLE_DEVICES=2 python taskluncher-20240201.py --lr 0.001 --advtrain_detector --onlyadvmail --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 20 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 4096 --rec_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/reconnaissance-detector-acc-0.9648.h5 --inf_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/infection-detector-acc-0.9480.h5 --att_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230728/action-detector-acc-0.9773.h5 --seq2seq_model_path /home/huan1932/ARIoTEDef-Actual/result/savemodel/20230729/infection-seq2seq-seed0-acc-0.9337.h5 > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-maxiter-20-epoch-40-batch-4096-lr0.001-onlyadvmail-advtrain_adv-seed-0-test-fit.log 2>&1




# CUDA_VISIBLE_DEVICES=2 python taskluncher-20240201.py --advtrain_detector --onlyadvmail --advtrain_adv_cle --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 10 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 4096  > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-maxiter-10-epoch-40-batch-4096-onlyadvmail-advtrain_adv_cle-seed-0.log 2>&1



# CUDA_VISIBLE_DEVICES=1 python taskluncher-20240201.py --advtrain_detector --onlyadvmail --advtrain_adv_cle --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 10 --patience 50 --timesteps 1 --seed 0 --ps_epochs 40 --batchsize 32  > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-maxiter-10-epoch-40-batch-32-onlyadvmail-advtrain_adv_cle-seed-0-fit.log 2>&1


# ing
CUDA_VISIBLE_DEVICES=2 python taskluncher-20240201.py --advtrain_detector --onlyadvmail --advtrain_adv_cle --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 10 --patience 50 --timesteps 1 --seed 2 --ps_epochs 40 --batchsize 4096  > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-maxiter-10-epoch-40-batch-4096-onlyadvmail-advtrain_adv_cle-seed-2-fit.log 2>&1

# ing
CUDA_VISIBLE_DEVICES=1 python taskluncher-20240201.py --advtrain_detector --onlyadvmail --advtrain_adv_cle --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 10 --patience 50 --timesteps 1 --seed 1 --ps_epochs 40 --batchsize 4096  > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-maxiter-10-epoch-40-batch-4096-onlyadvmail-advtrain_adv_cle-seed-1-fit.log 2>&1


# next 1
CUDA_VISIBLE_DEVICES=2 python taskluncher-20240201.py --advtrain_detector --onlyadvmail --advtrain_adv_cle --attack pgd --eps 1.0 --eps_step 0.5 --max_iter 10 --patience 50 --timesteps 1 --seed 1 --ps_epochs 40 --batchsize 32  > /home/huan1932/ARIoTEDef-Actual/log/20240201/advtrain-infection-detector-maxiter-10-epoch-40-batch-32-onlyadvmail-advtrain_adv_cle-seed-1-fit.log 2>&1