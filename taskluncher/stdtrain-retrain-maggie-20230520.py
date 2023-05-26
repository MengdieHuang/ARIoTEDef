""" 
written by Zilin Shen, Daniel de Mello, and Mengdie Huang
Date: May 8, 2023
"""

import os
os.environ['TF_NUMA_NODES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置日志级别为 WARNING
import numpy as np
from seq2seq.utils import print_header, get_events
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from argsparse import get_args
from savedir import set_exp_result_dir
from data.dataload import loadnpydata
from data.normalize import normalize_multistep_dataset
from models.createmodel import init_psdetector
from models.createmodel import init_seq2seq
from utils.events import get_events_from_windows
from data.truncation import truncationdata

# -----------parse parameters-----------
print("\n")
args = get_args()
seq2seq_config = {"sequence_length": args.sequence_length, 
                  "permute_truncated": args.permute_truncated,
                  "use_prob_embedding": args.use_prob_embedding,
                  "rv": args.rv
                  }   
print("args:",args)

# -----------set save path-----------
exp_result_dir = set_exp_result_dir(args)
os.makedirs(exp_result_dir, exist_ok=True)
print("exp_result_dir:",exp_result_dir)

# -----------get the preprocessed training and testing saved as .npy files
multistep_dataset = loadnpydata()

truncate_multistep_dataset = truncationdata(multistep_dataset)
    
norm_multistep_dataset = normalize_multistep_dataset(truncate_multistep_dataset)

# ----------------create per-step detectors----------------------
reconnaissance_detector, infection_detector, attack_detector = init_psdetector(norm_multistep_dataset, args)


# ----------------train per-step detectors----------------------
print_header("Train Per-Step Detector")
for detector in [reconnaissance_detector, infection_detector, attack_detector]:
    
    print(f">>>>>>>> Training {detector.modelname} >>>>>>>>")    
    detector.stdtrain(timesteps=args.timesteps, exp_result_dir=exp_result_dir)
    
    print(f">>>>>>>> Evaluate {detector.modelname} on clean test data")
    test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1 = detector.test(testset_x=detector.dataset['test'][0], testset_y=detector.dataset['test'][1],timesteps=args.timesteps, exp_result_dir=exp_result_dir)
    
    metrics_dic = { 
                   'model': detector.modelname,
                   'clean test Accuracy': test_acc,
                   'clean test Loss': test_los,
                   'clean test TP': test_TP,
                   'clean test FP': test_FP,
                   'clean test TN': test_TN,
                   'clean test FN': test_FN,
                   'clean test Recall': test_recall,
                   'clean test Precision': test_precision,
                   'clean test FPR': test_FPR,
                   'clean test F1': test_F1,
                }
     
    print(f"{detector.modelname} metrics_dic:\n {metrics_dic}")   
 
# ----------------adversarial attack vanilla per-step detectors----------------------
print_header("Adversarial Attack Vanilla Per-Step Detector")
for detector in [reconnaissance_detector, infection_detector, attack_detector]:
   
    # generate adversarial mailicious testset
    print(f"Generate only adversarial mailicious exapmples based white-box {detector.modelname}")
    
    adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps, exp_result_dir=exp_result_dir)
        
    adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_FPR, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps, exp_result_dir=exp_result_dir)
    
    adv_metrics_dic = { 
                   'model': detector.modelname,
                   'adv test Accuracy': adv_test_acc,
                   'adv test Loss': adv_test_los,
                   'adv test TP': adv_test_TP,
                   'adv test FP': adv_test_FP,
                   'adv test TN': adv_test_TN,
                   'adv test FN': adv_test_FN,
                   'adv test Recall': adv_test_recall,
                   'adv test Precision': adv_test_precision,
                   'adv test FPR': adv_test_FPR,
                   'adv test F1': adv_test_F1,
                }
     
    print(f"{detector.modelname} adv_metrics_dic:\n {adv_metrics_dic}")           
            
# ----------------create seq2seq----------------------
print_header("Train Infection Seq2Seq")
infection_seq2seq = init_seq2seq(norm_multistep_dataset, args)

# ----------------train seq2seq----------------------
for seq2seq in [infection_seq2seq]:
    print(f">>>>>>>> Training {seq2seq.modelname} >>>>>>>>")    
    
    import math
 
    # create trainset
    cle_train_windows_x = seq2seq.dataset['train'][0]
    cle_train_windows_y = seq2seq.dataset['train'][1]
    print("cle_train_windows_x.shape:",cle_train_windows_x.shape)
    print("cle_train_windows_y.shape:",cle_train_windows_y.shape)
    cle_train_windows_x = cle_train_windows_x.reshape((cle_train_windows_x.shape[0], args.timesteps, int(math.ceil(cle_train_windows_x.shape[1] / args.timesteps))))
    print("cle_train_windows_x.shape:",cle_train_windows_x.shape)
    
    seq2seq_train_events = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_train_windows_x)
    print("seq2seq_train_events.shape:",seq2seq_train_events.shape)
    """ 
    seq2seq_train_events.shape: (19808, 4)
    """    
    
    seq2seq.stdtrain(events=seq2seq_train_events, labels=cle_train_windows_y, exp_result_dir=exp_result_dir)
    
    
    # raise Exception("maggie")
    print(f">>>>>>>> Evaluate {seq2seq.modelname} on clean test data")
    
    # create testset
    cle_test_windows_x = seq2seq.dataset['test'][0]
    cle_test_windows_y = seq2seq.dataset['test'][1]
    print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
    print("cle_test_windows_y.shape:",cle_test_windows_y.shape)
    cle_test_windows_x = cle_test_windows_x.reshape((cle_test_windows_x.shape[0], args.timesteps, int(math.ceil(cle_test_windows_x.shape[1] / args.timesteps))))
    print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
    
    seq2seq_test_events = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_test_windows_x)
    print("seq2seq_test_events.shape:",seq2seq_test_events.shape)
    """ 

    """        
    test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1 = seq2seq.test(testset_x=seq2seq_test_events, testset_y=cle_test_windows_y, exp_result_dir=exp_result_dir)
        
# ----------------retrain vanilla per-step detectors----------------------
print_header("Retrain Vanilla Infection Detector")
for detector in [infection_detector]:

    print_header("Retrain {} detector".format(detector.modelname))
    
    
    
    
    
    
# cle_testset_x = detector.dataset['test'][0]
# cle_testset_y = detector.dataset['test'][1]
# cle_trainset_x = detector.dataset['train'][0]
# cle_trainset_y = detector.dataset['train'][1]
# print("cle_testset_x.shape:",cle_testset_x.shape)
# print("cle_testset_y.shape:",cle_testset_y.shape)
# print("cle_trainset_x.shape:",cle_trainset_x.shape)
# print("cle_trainset_y.shape:",cle_trainset_y.shape)            

# """
# cle_testset_x.shape: (4233, 41)
# cle_testset_y.shape: (4233,)
# cle_trainset_x.shape: (19818, 41)
# cle_trainset_y.shape: (19818,)
# """  

# # init seq2seq
# # seq2seq = Seq2seqAttention('seq2seq')              

# for r in range(args.relabel_rounds):
        # print("\n round_index:",r)
        # curround_exp_result_dir = os.path.join(exp_result_dir,f'round-{r}')
         
        # print_header("{}-th train seq2seq".format(r))
        # #get events
        # train_events = get_events(ps_attack, ps_recon, ps_infec, cle_trainset_x)                
        # # train seq2seq using events from infection training set
        # seq2seq.learning(train_events, cle_trainset_y, seq2seq_config)
        
        
        # # on clean exampels
        # # -----------------seq2seq stage----------------------        
        # print_header("{}-th trained seq2seq predict test events".format(r))
        # test_events = get_events(ps_attack, ps_recon, ps_infec, cle_testset_x) 

        # #get seq2seq tagged events in training set
        # test_events_preds, tagged_seq2seq = seq2seq.analysis(test_events, cle_testset_y, seq2seq_config)
        # """ 
        # test_events_preds is the probality list of samples predicted as infection by seq2seq
        # tagged_seq2seq is the indx list of samples in test set predicted as infection by seq2seq        
        # """

        # # -----------------per-step detector stage----------------------
        # print_header("{}-th trained per-step detector predict test windows".format(r))        
        # # get per-step infection detector tagged windows
        # preds_detector = detector.predict(cle_testset_x, kind='')
        # preds_detector = np.array(preds_detector).squeeze()
        # # print('preds_detector.shape:', preds_detector.shape)
        # # preds ps infec shape is (19818,)
        
        # # dataset A
        # tagged_detector = []
        # for idx, pred in enumerate(preds_detector):
        #     if pred>0.5:    # predicted label = infection
        #         tagged_detector.append(idx)
        # """ 
        # tagged_detector is the indx list of samples in test set predicted as infection by infection detector
        # """       
        # # -----------------relabeling---------------------
        # #strategy 1
        # # dataset A tagged_detector
        
        # # dataset B tagged_seq2seq
        
        # # dataset C A/B
        
        # retrain_pos = []
        # retrain_neg = []

        # for idx in tagged_detector:         #   all samples in set A
        #     if idx in tagged_seq2seq:       #   all samples in set B
        #         retrain_pos.append(idx)     #   infection array
        #     else:                           #   all samples in set C
        #         retrain_neg.append(idx)     #   benign array    


        # # extrac relabeled test sampels in clean infection test
        # ori_cle_testset_x = copy.deepcopy(cle_testset_x)
        # ori_cle_testset_y = copy.deepcopy(cle_testset_y)        
        
        # retrain_cle_mal_testset_x=[]
        # retrain_cle_mal_testset_y=[]
        # retrain_cle_ben_testset_x=[]
        # retrain_cle_ben_testset_y=[]        
        # for idx, l in enumerate(ori_cle_testset_y): # 遍历test set
        #     if idx in retrain_pos:      # infection 样本                
        #         retrain_cle_mal_testset_x.append(ori_cle_testset_x[idx])
        #         retrain_cle_mal_testset_y.append(1)
        #         # retrain_cle_mal_testset_y.append(ori_cle_testset_y[idx])
                
        #     elif idx in retrain_neg:    # benign 样本
        #         retrain_cle_ben_testset_x.append(ori_cle_testset_x[idx])
        #         retrain_cle_ben_testset_y.append(0)
        #         # retrain_cle_ben_testset_y.append(ori_cle_testset_y[idx])
                
        # retrain_cle_mal_testset_x = np.array(retrain_cle_mal_testset_x)        
        # retrain_cle_mal_testset_y = np.array(retrain_cle_mal_testset_y)        
        # retrain_cle_ben_testset_x = np.array(retrain_cle_ben_testset_x)        
        # retrain_cle_ben_testset_y = np.array(retrain_cle_ben_testset_y)        
                
        # print("retrain_cle_mal_testset_x.shape:",retrain_cle_mal_testset_x.shape)
        # print("retrain_cle_mal_testset_y.shape:",retrain_cle_mal_testset_y.shape)
        # print("retrain_cle_mal_testset_y:",retrain_cle_mal_testset_y)
        # print("retrain_cle_ben_testset_x.shape:",retrain_cle_ben_testset_x.shape)
        # print("retrain_cle_ben_testset_y.shape:",retrain_cle_ben_testset_y.shape)    
        # print("retrain_cle_ben_testset_y:",retrain_cle_ben_testset_y)
        
        # retrain_cle_testset_x = np.concatenate((retrain_cle_mal_testset_x,retrain_cle_ben_testset_x))
        # retrain_cle_testset_y = np.concatenate((retrain_cle_mal_testset_y,retrain_cle_ben_testset_y))
        # print("retrain_cle_testset_x.shape:",retrain_cle_testset_x.shape)
        # print("retrain_cle_testset_y.shape:",retrain_cle_testset_y.shape)
        # print("retrain_cle_testset_y:",retrain_cle_testset_y)
                
        # """ 
        # retrain_cle_mal_testset_x.shape: (380, 41)
        # retrain_cle_mal_testset_y.shape: (380,)
        # retrain_cle_ben_testset_x.shape: (19, 41)
        # retrain_cle_ben_testset_y.shape: (19,)
        # retrain_cle_testset_x.shape: (399, 41)
        # retrain_cle_testset_y.shape: (399,)
        # """                
                
                
                
                
                
        # # on adversarial exampels
        # print_header("Measureing {} detector performance on clean test data".format(detector.name))
        # _, _, cle_metrics_dict_ps = detector.detection(dataset=cle_testset_x, label=cle_testset_y, kind='')
        # print("Metrics on clean testset: \n", cle_metrics_dict_ps)

        # # generate adversarial infection testset
        # print_header("Generate only adversarial mailicious exapmples based white-box {} detector with clean test data".format(detector.name))
        # adv_test_examples, adv_test_labels = detector.advgenerate_onlymail(cle_testset_x=cle_testset_x, cle_testset_y=cle_testset_y, kind='', args=args)
            
        # print("adv_test_examples.shape:",adv_test_examples.shape)
        # print("adv_test_labels.shape:",adv_test_labels.shape)
        # # print("adv_test_labels:",adv_test_labels)
        # """ 
        # adv_test_examples.shape: (318, 41)
        # adv_test_labels.shape: (318,)
        # """
        
        # # adversarial evasion attack vanilla infection detector
        # print_header("Measureing {} detector performance on adversarial test data".format(detector.name))
        # _, _, adv_metrics_dict_ps = detector.detection(adv_test_examples, adv_test_labels, kind='')
        # print("Metrics on adversarial testset: \n", adv_metrics_dict_ps)
        
            
            
        # adv_testset_x = adv_test_examples
        # adv_testset_y = adv_test_labels
        
        # # -----------------seq2seq stage----------------------        
        # print_header("{}-th trained seq2seq predict adversarial malicious test events".format(r))
        # test_events = get_events(ps_attack, ps_recon, ps_infec, adv_testset_x) 

        # #get seq2seq tagged events in training set
        # test_events_preds, tagged_seq2seq = seq2seq.analysis(test_events, adv_testset_y, seq2seq_config)
        # """ 
        # test_events_preds is the probality list of samples predicted as infection by seq2seq
        # tagged_seq2seq is the indx list of samples in test set predicted as infection by seq2seq        
        # """

        # # -----------------per-step detector stage----------------------
        # print_header("{}-th trained per-step detector predict adversarial malicious test windows".format(r))        
        # # get per-step infection detector tagged windows
        # preds_detector = detector.predict(adv_testset_x, kind='')
        # preds_detector = np.array(preds_detector).squeeze()
        # print('preds_detector.shape:', preds_detector.shape)
        # # preds_detector.shape: (318,)
        
        # # dataset A
        # tagged_detector = []
        # for idx, pred in enumerate(preds_detector):
        #     if pred>0.5:    # predicted label = infection
        #         tagged_detector.append(idx)
        # """ 
        # tagged_detector is the indx list of samples in test set predicted as infection by infection detector
        # """       
        # # -----------------relabeling---------------------
        # #strategy 1
        # # dataset A tagged_detector
        
        # # dataset B tagged_seq2seq
        
        # # dataset C A/B
        
        # retrain_pos = []
        # retrain_neg = []

        # for idx in tagged_detector:         #   all samples in set A
        #     if idx in tagged_seq2seq:       #   all samples in set B
        #         retrain_pos.append(idx)     #   infection array
        #     else:                           #   all samples in set C
        #         retrain_neg.append(idx)     #   benign array    
                                

        # # extrac relabeled test sampels in adversarial infection test
        # ori_adv_testset_x = copy.deepcopy(adv_testset_x)
        # ori_adv_testset_y = copy.deepcopy(adv_testset_y)        
        
        # retrain_adv_mal_testset_x=[]
        # retrain_adv_mal_testset_y=[]
        # retrain_adv_ben_testset_x=[]
        # retrain_adv_ben_testset_y=[]        
        # for idx, l in enumerate(ori_adv_testset_y): # 遍历test set
        #     if idx in retrain_pos:      # infection 样本                
        #         retrain_adv_mal_testset_x.append(ori_adv_testset_x[idx])
        #         retrain_adv_mal_testset_y.append(ori_adv_testset_y[idx])
                
        #     elif idx in retrain_neg:    # benign 样本
        #         retrain_adv_ben_testset_x.append(ori_adv_testset_x[idx])
        #         retrain_adv_ben_testset_y.append(ori_adv_testset_y[idx])
                
        # retrain_adv_mal_testset_x = np.array(retrain_adv_mal_testset_x)        
        # retrain_adv_mal_testset_y = np.array(retrain_adv_mal_testset_y)        
        # retrain_adv_ben_testset_x = np.array(retrain_adv_ben_testset_x)        
        # retrain_adv_ben_testset_y = np.array(retrain_adv_ben_testset_y)        
                
        # print("retrain_adv_mal_testset_x.shape:",retrain_adv_mal_testset_x.shape)
        # print("retrain_adv_mal_testset_y.shape:",retrain_adv_mal_testset_y.shape)
        # print("retrain_adv_ben_testset_x.shape:",retrain_adv_ben_testset_x.shape)
        # print("retrain_adv_ben_testset_y.shape:",retrain_adv_ben_testset_y.shape)
        
        
        # retrain_testset_x = np.concatenate((retrain_cle_testset_x,retrain_adv_mal_testset_x))
        # retrain_testset_y = np.concatenate((retrain_cle_testset_y,retrain_adv_mal_testset_y))
        # print("retrain_testset_x.shape:",retrain_testset_x.shape)
        # print("retrain_testset_y.shape:",retrain_testset_y.shape)
        # print("retrain_testset_y:",retrain_testset_y)
                
                        
        # raise Exception("maggie stop here!!!!!!")    