""" 
written by Zilin Shen, Daniel de Mello, and Mengdie Huang
Date: May 8, 2023
"""

import os
os.environ['TF_NUMA_NODES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # set log level WARNING
import numpy as np
from seq2seq.utils import print_header, get_events
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from utils.argsparse import get_args
from savedir import set_exp_result_dir
from data.dataload import loadnpydata
from data.normalize import normalize_multistep_dataset
from models.createmodel import init_psdetector
from models.createmodel import init_seq2seq
from utils.events import get_events_from_windows
from data.truncation import truncationdata
import copy

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
    stdtrain_exp_result_dir = os.path.join(exp_result_dir,f'stdtrain-psdetector')
    os.makedirs(stdtrain_exp_result_dir, exist_ok=True)

    detector.stdtrain(timesteps=args.timesteps, exp_result_dir=stdtrain_exp_result_dir)
    
    print(f">>>>>>>> Evaluate {detector.modelname} on clean test data")
    test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1 = detector.test(testset_x=detector.dataset['test'][0], testset_y=detector.dataset['test'][1],timesteps=args.timesteps, exp_result_dir=stdtrain_exp_result_dir)
    
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
    print(f"Generate adversarial mailicious exapmples based white-box {detector.modelname}")

    adv_exp_result_dir = os.path.join(exp_result_dir,f'advattack')
    os.makedirs(adv_exp_result_dir, exist_ok=True)
        
    adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps, exp_result_dir=adv_exp_result_dir)
    
    print("adv_testset_x.shape:",adv_testset_x.shape)    
    print("adv_testset_y.shape:",adv_testset_y.shape)    

    adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_FPR, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps, exp_result_dir=adv_exp_result_dir)
    
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
    
    stdtrain_exp_result_dir = os.path.join(exp_result_dir,f'stdtrain-seq2seq')
    os.makedirs(stdtrain_exp_result_dir, exist_ok=True)
    
    seq2seq.stdtrain(events=seq2seq_train_events, labels=cle_train_windows_y, exp_result_dir=stdtrain_exp_result_dir)
        
    print(f">>>>>>>> Evaluate {seq2seq.modelname} on clean test data")
    
    # create testset
    cle_test_windows_x = seq2seq.dataset['test'][0]
    cle_test_windows_y = seq2seq.dataset['test'][1]
    print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
    print("cle_test_windows_y.shape:",cle_test_windows_y.shape)
    cle_test_windows_x = cle_test_windows_x.reshape((cle_test_windows_x.shape[0], args.timesteps, int(math.ceil(cle_test_windows_x.shape[1] / args.timesteps))))
    print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
    
    """ 
    cle_test_windows_x.shape: (4224, 41)
    cle_test_windows_y.shape: (4224,)
    cle_test_windows_x.shape: (4224, 1, 41)
    """
    seq2seq_test_events = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_test_windows_x)
    print("seq2seq_test_events.shape:",seq2seq_test_events.shape)
    """ 
    seq2seq_test_events.shape: (4224, 4)
    """        
    test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1 = seq2seq.test(testset_x=seq2seq_test_events, testset_y=cle_test_windows_y, exp_result_dir=stdtrain_exp_result_dir)
    
    metrics_dic = { 
                   'model': seq2seq.modelname,
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
     
    print(f"{seq2seq.modelname} metrics_dic:\n {metrics_dic}")       
        
# ----------------retrain vanilla per-step detectors----------------------
print_header("Retrain Vanilla Infection Detector")

for detector in [infection_detector]:

    for seq2seq in [infection_seq2seq]:
        
        vanillia_detector = copy.deepcopy(detector)
        
        print_header(f"Retrain {detector.modelname} detector using {infection_seq2seq.modelname}")

        test_acc_list = [] 
        test_los_list = []
        test_TP_list = []
        test_FP_list = []
        test_TN_list = []
        test_FN_list = []
        test_recall_list = []
        test_precision_list = []
        test_FPR_list = []
        test_F1_list = []
        cost_time_list =[]


        adv_test_acc_list = [] 
        adv_test_los_list = []
        adv_test_TN_list = []
        adv_test_FN_list = []
        adv_test_recall_list = []
        adv_test_precision_list = []
        adv_test_F1_list = []
        
        #---------------evaluate vanilia detector---------------
        print(f">>>>>>>> Evaluate vanillia {detector.modelname} on clean test data >>>>>>>>")
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
        
        print(f"vanillia {detector.modelname} metrics_dic:\n {metrics_dic}")  

        test_acc_list.append(test_acc)
        test_los_list.append(test_los)
        test_TP_list.append(test_TP)
        test_FP_list.append(test_FP)
        test_TN_list.append(test_TN)
        test_FN_list.append(test_FN)
        test_recall_list.append(test_recall)
        test_precision_list.append(test_precision)
        test_FPR_list.append(test_FPR)
        test_F1_list.append(test_F1)
        cost_time_list.append(0)        
                
        print(f">>>>>>>> Evaluate vanillia {detector.modelname} on adversarial test data >>>>>>>>")

        adv_exp_result_dir = os.path.join(exp_result_dir,f'advattack')
        os.makedirs(adv_exp_result_dir, exist_ok=True)
            
        adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps, exp_result_dir=adv_exp_result_dir)
        
        print("adv_testset_x.shape:",adv_testset_x.shape)    
        print("adv_testset_y.shape:",adv_testset_y.shape)    
        
        adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_FPR, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps, exp_result_dir=adv_exp_result_dir)
        
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
        
        print(f"vanillia {detector.modelname} adv_metrics_dic:\n {adv_metrics_dic}")    

        adv_test_acc_list.append(adv_test_acc)
        adv_test_los_list.append(adv_test_los)
        adv_test_TN_list.append(adv_test_TN)
        adv_test_FN_list.append(adv_test_FN)
        adv_test_recall_list.append(adv_test_recall)
        adv_test_precision_list.append(adv_test_precision)
        adv_test_F1_list.append(adv_test_F1)
        
        #-----------------------------------------------------    
                
        for r in range(args.relabel_rounds):
            print(f">>>>>>>>>>>>>> {r+1} round retraining: >>>>>>>>>>>>>> ")    
            curround_exp_result_dir = os.path.join(exp_result_dir,f'round-{r+1}')
            os.makedirs(curround_exp_result_dir, exist_ok=True)

            print("args.retrainset_mode:",args.retrainset_mode)

            if args.retrainset_mode == 'olds1':

                # create testset
                cle_test_windows_x = detector.dataset['test'][0]
                cle_test_windows_y = detector.dataset['test'][1]
                
                # print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
                # print("cle_test_windows_y.shape:",cle_test_windows_y.shape)
                
                cle_test_windows_x = cle_test_windows_x.reshape((cle_test_windows_x.shape[0], args.timesteps, int(math.ceil(cle_test_windows_x.shape[1] / args.timesteps))))
                
                # print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
                
                """ 
                cle_test_windows_x.shape: (4224, 41)
                cle_test_windows_y.shape: (4224,)
                cle_test_windows_x.shape: (4224, 1, 41)
                """
                
                seq2seq_test_events = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_test_windows_x)
                # print("seq2seq_test_events.shape:",seq2seq_test_events.shape)
                """ 
                seq2seq_test_events.shape: (4224, 4)
                """      

                seq2seq_tagged_events_probs, seq2seq_tagged_events_idxs = seq2seq.analysis(seq2seq_test_events, cle_test_windows_y, curround_exp_result_dir)
                """ 
                seq2seq_tagged_events_idxs is the indx list of samples in test set predicted as infection by infection seq2seq
                """ 
                            
                seq2seq_tagged_events_probs = np.array(seq2seq_tagged_events_probs)
                seq2seq_tagged_events_idxs = np.array(seq2seq_tagged_events_idxs)
                # print('seq2seq_tagged_events_probs.shape:', seq2seq_tagged_events_probs.shape)
                # print('seq2seq_tagged_events_idxs.shape:', seq2seq_tagged_events_idxs.shape)
                """ 
                seq2seq_tagged_events_probs.shape: (351,)
                seq2seq_tagged_events_idxs.shape: (351,)
                """
                
                detector_probs = detector.model.predict(cle_test_windows_x)
                detector_probs = np.array(detector_probs).squeeze()
                # print('detector_probs.shape:', detector_probs.shape)
        
                detector_tagged_windows_idxs = []
                detector_tagged_windows_probs = []
                
                for idx, pred in enumerate(detector_probs):
                    if pred>0.5:    # predicted label = infection
                        detector_tagged_windows_idxs.append(idx)
                        detector_tagged_windows_probs.append(pred)
                """ 
                detector_tagged_windows_idxs is the indx list of samples in test set predicted as infection by infection detector
                """       

                detector_tagged_windows_probs = np.array(detector_tagged_windows_probs)            
                detector_tagged_windows_idxs = np.array(detector_tagged_windows_idxs)
                # print('detector_tagged_windows_probs.shape:', detector_tagged_windows_probs.shape)
                # print('detector_tagged_windows_idxs.shape:', detector_tagged_windows_idxs.shape)
                
                """ 
                detector_tagged_windows_probs.shape: (261,)
                detector_tagged_windows_idxs.shape: (261,)
                """
                
                            
                # -----------------relabeling---------------------
                #strategy 1
                # dataset A detector_tagged_windows_idxs
                # dataset B seq2seq_tagged_events_idxs
                # dataset C A/B
                
                retrain_pos = []
                retrain_neg = []

                # for idx in seq2seq_tagged_events_idxs:          #   all samples in set B
                #     retrain_pos.append(idx)                     #   infection array
                # for idx in detector_tagged_windows_idxs:        #   all samples in set A
                #     if idx not in seq2seq_tagged_events_idxs:   #   all samples in set C=A/B
                #         retrain_neg.append(idx)                 #   benign array    

                for idx in detector_tagged_windows_idxs:            #   all samples in set A
                    if idx in seq2seq_tagged_events_idxs:           #   all samples in set B
                        retrain_pos.append(idx)
                    else:                                           #   all samples in set C=A/B
                        retrain_neg.append(idx)
            

                print(f">>>>>>>> Prepare {r+1} round Retraining clean dataset >>>>>>>>")    

                # extract relabeled test sampels in clean infection test
                ori_cle_testset_x = copy.deepcopy(detector.dataset['test'][0])
                ori_cle_testset_y = copy.deepcopy(detector.dataset['test'][1])        
                
                retrain_cle_mal_testset_x=[]
                retrain_cle_mal_testset_y=[]
                retrain_cle_ben_testset_x=[]
                retrain_cle_ben_testset_y=[]        
                for idx, l in enumerate(ori_cle_testset_y): # 遍历test set
                    if idx in retrain_pos:                   
                        retrain_cle_mal_testset_x.append(ori_cle_testset_x[idx])
                        retrain_cle_mal_testset_y.append(1)
                        
                    elif idx in retrain_neg:            
                        retrain_cle_ben_testset_x.append(ori_cle_testset_x[idx])
                        retrain_cle_ben_testset_y.append(0)
                        
                retrain_cle_mal_testset_x = np.array(retrain_cle_mal_testset_x)        
                retrain_cle_mal_testset_y = np.array(retrain_cle_mal_testset_y)        
                retrain_cle_ben_testset_x = np.array(retrain_cle_ben_testset_x)        
                retrain_cle_ben_testset_y = np.array(retrain_cle_ben_testset_y)        
                        
                # print("retrain_cle_mal_testset_x.shape:",retrain_cle_mal_testset_x.shape)
                # print("retrain_cle_mal_testset_y.shape:",retrain_cle_mal_testset_y.shape)
                # print("retrain_cle_mal_testset_y:",retrain_cle_mal_testset_y)
                """ 
                retrain_cle_mal_testset_x.shape: (3833, 41)
                retrain_cle_mal_testset_y.shape: (3833,)
                """
                
                # print("retrain_cle_ben_testset_x.shape:",retrain_cle_ben_testset_x.shape)
                # print("retrain_cle_ben_testset_y.shape:",retrain_cle_ben_testset_y.shape)    
                # print("retrain_cle_ben_testset_y:",retrain_cle_ben_testset_y)
                
                """ 
                retrain_cle_ben_testset_x.shape: (19, 41)
                retrain_cle_ben_testset_y.shape: (19,)
                """
                
                retrain_cle_testset_x = np.concatenate((retrain_cle_mal_testset_x,retrain_cle_ben_testset_x))
                retrain_cle_testset_y = np.concatenate((retrain_cle_mal_testset_y,retrain_cle_ben_testset_y))
                print("retrain_cle_testset_x.shape:",retrain_cle_testset_x.shape)
                print("retrain_cle_testset_y.shape:",retrain_cle_testset_y.shape)
                # print("retrain_cle_testset_y:",retrain_cle_testset_y)
                        
                """ 
                retrain_cle_testset_x.shape: (3852, 41)
                retrain_cle_testset_y.shape: (3852,)
                """     

                # create trainset
                # retrain_testset_len = len(retrain_cle_testset_x)
                # print("retrain_testset_len:",retrain_testset_len)
                # ori_cle_trainset_x = copy.deepcopy(detector.dataset['train'][0])
                # ori_cle_trainet_y = copy.deepcopy(detector.dataset['train'][1]) 
                                                    
                # retrain_cle_trainset_x = ori_cle_trainset_x[retrain_testset_len*r:retrain_testset_len*(r+1)]
                # retrain_cle_trainset_y = ori_cle_trainet_y[retrain_testset_len*r:retrain_testset_len*(r+1)]
                # print("retrain_cle_trainset_x.shape:",retrain_cle_trainset_x.shape)
                # print("retrain_cle_trainset_y.shape:",retrain_cle_trainset_y.shape)

                # """ 
                # retrain_testset_len: 3852
                # retrain_cle_trainset_x.shape: (3852, 41)
                
                # """
                retrain_cle_set_x = np.concatenate((retrain_cle_testset_x,detector.dataset['train'][0]))
                retrain_cle_set_y = np.concatenate((retrain_cle_testset_y,detector.dataset['train'][1]))            
                print("retrain_cle_set_x.shape:",retrain_cle_set_x.shape)
                print("retrain_cle_set_y.shape:",retrain_cle_set_y.shape)            
                # print("retrain_cle_set_y:",retrain_cle_set_y)
                
                """ 
                retrain_cle_set_x.shape: (7704, 41)

                """
                #-----------------------------------------------
                retrainset_x = retrain_cle_set_x
                retrainset_y = retrain_cle_set_y
            
            if args.retrainset_mode == 'adv':
                print(f">>>>>>>> Prepare {r+1} round Retraining adversarial dataset >>>>>>>>")    
            
                adv_exp_result_dir = os.path.join(curround_exp_result_dir,f'advattack')
                os.makedirs(adv_exp_result_dir, exist_ok=True)
                    
                retrain_adv_testset_x, retrain_adv_testset_y = detector.generate_advmail(timesteps=args.timesteps, exp_result_dir=adv_exp_result_dir)
                
                print("retrain_adv_testset_x.shape:",retrain_adv_testset_x.shape)    
                print("retrain_adv_testset_y.shape:",retrain_adv_testset_y.shape)    
                

                print(f">>>>>>>> Prepare {r+1} round Retraining clean dataset >>>>>>>>")    
                
                retrain_advtestset_len = len(retrain_adv_testset_x)
                print("retrain_advtestset_len:",retrain_advtestset_len)
                ori_cle_trainset_x = copy.deepcopy(detector.dataset['train'][0])
                ori_cle_trainet_y = copy.deepcopy(detector.dataset['train'][1]) 
                                                    
                retrain_cle_trainset_x = ori_cle_trainset_x[retrain_advtestset_len*r:retrain_advtestset_len*(r+1)]
                retrain_cle_trainset_y = ori_cle_trainet_y[retrain_advtestset_len*r:retrain_advtestset_len*(r+1)]
                print("retrain_cle_trainset_x.shape:",retrain_cle_trainset_x.shape)
                print("retrain_cle_trainset_y.shape:",retrain_cle_trainset_y.shape)

                retrainset_x = np.concatenate((retrain_adv_testset_x,retrain_cle_trainset_x))
                retrainset_y = np.concatenate((retrain_adv_testset_y,retrain_cle_trainset_y))            
                print("retrainset_x.shape:",retrainset_x.shape)
                print("retrainset_y.shape:",retrainset_y.shape) 
                
            if args.retrainset_mode == 'advs1':

                # create testset
                print(f">>>>>>>> Prepare {r+1} round Retraining adversarial dataset >>>>>>>>")    
            
                adv_exp_result_dir = os.path.join(curround_exp_result_dir,f'advattack')
                os.makedirs(adv_exp_result_dir, exist_ok=True)
                    
                adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps, exp_result_dir=adv_exp_result_dir)
                
                print("adv_testset_x.shape:",adv_testset_x.shape)    
                print("adv_testset_y.shape:",adv_testset_y.shape)    
                                
                
                cle_test_x = detector.dataset['test'][0]
                cle_test_y = detector.dataset['test'][1]
                
                print("cle_test_x.shape:",cle_test_x.shape)
                print("cle_test_y.shape:",cle_test_y.shape)
                
                test_x = np.concatenate((adv_testset_x,cle_test_x))
                test_y = np.concatenate((adv_testset_y,cle_test_y))            
                print("test_x.shape:",test_x.shape)
                print("test_y.shape:",test_y.shape)                 
                

                test_windows_x = copy.deepcopy(test_x)
                test_windows_y = copy.deepcopy(test_y)           
                print("test_windows_x.shape:",test_windows_x.shape)
                print("test_windows_y.shape:",test_windows_y.shape) 
                
                                
                test_windows_x = test_windows_x.reshape((test_windows_x.shape[0], args.timesteps, int(math.ceil(test_windows_x.shape[1] / args.timesteps))))
                
                print("test_windows_x.shape:",test_windows_x.shape)
                
                """ 
       
                """
                
                seq2seq_test_events = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, test_windows_x)
                # print("seq2seq_test_events.shape:",seq2seq_test_events.shape)
                """ 
                seq2seq_test_events.shape: (4224, 4)
                """      

                seq2seq_tagged_events_probs, seq2seq_tagged_events_idxs = seq2seq.analysis(seq2seq_test_events, test_windows_y, curround_exp_result_dir)
                """ 
                seq2seq_tagged_events_idxs is the indx list of samples in test set predicted as infection by infection seq2seq
                """ 
                            
                seq2seq_tagged_events_probs = np.array(seq2seq_tagged_events_probs)
                seq2seq_tagged_events_idxs = np.array(seq2seq_tagged_events_idxs)
                # print('seq2seq_tagged_events_probs.shape:', seq2seq_tagged_events_probs.shape)
                # print('seq2seq_tagged_events_idxs.shape:', seq2seq_tagged_events_idxs.shape)
                """ 
                seq2seq_tagged_events_probs.shape: (351,)
                seq2seq_tagged_events_idxs.shape: (351,)
                """
                
                detector_probs = detector.model.predict(test_windows_x)
                detector_probs = np.array(detector_probs).squeeze()
                # print('detector_probs.shape:', detector_probs.shape)
        
                detector_tagged_windows_idxs = []
                detector_tagged_windows_probs = []
                
                for idx, pred in enumerate(detector_probs):
                    if pred>0.5:    # predicted label = infection
                        detector_tagged_windows_idxs.append(idx)
                        detector_tagged_windows_probs.append(pred)
                """ 
                detector_tagged_windows_idxs is the indx list of samples in test set predicted as infection by infection detector
                """       

                detector_tagged_windows_probs = np.array(detector_tagged_windows_probs)            
                detector_tagged_windows_idxs = np.array(detector_tagged_windows_idxs)
                # print('detector_tagged_windows_probs.shape:', detector_tagged_windows_probs.shape)
                # print('detector_tagged_windows_idxs.shape:', detector_tagged_windows_idxs.shape)
                
                """ 
                detector_tagged_windows_probs.shape: (261,)
                detector_tagged_windows_idxs.shape: (261,)
                """
                
                            
                # -----------------relabeling---------------------
                #strategy 1
                # dataset A detector_tagged_windows_idxs
                # dataset B seq2seq_tagged_events_idxs
                # dataset C A/B
                
                retrain_pos = []
                retrain_neg = []

                # for idx in seq2seq_tagged_events_idxs:          #   all samples in set B
                #     retrain_pos.append(idx)                     #   infection array
                # for idx in detector_tagged_windows_idxs:        #   all samples in set A
                #     if idx not in seq2seq_tagged_events_idxs:   #   all samples in set C=A/B
                #         retrain_neg.append(idx)                 #   benign array    
                
                for idx in detector_tagged_windows_idxs:
                    if idx in seq2seq_tagged_events_idxs:
                        retrain_pos.append(idx)
                    else:
                        retrain_neg.append(idx)                


                print(f">>>>>>>> Prepare {r+1} round Retraining clean dataset >>>>>>>>")    

                # extract relabeled test sampels in clean infection test
                ori_testset_x = test_x
                ori_testset_y = test_y   
                
                retrain_mal_testset_x=[]
                retrain_mal_testset_y=[]
                retrain_ben_testset_x=[]
                retrain_ben_testset_y=[]        
                for idx, l in enumerate(ori_testset_y): # 遍历test set
                    if idx in retrain_pos:                   
                        retrain_mal_testset_x.append(ori_testset_x[idx])
                        retrain_mal_testset_y.append(1)
                        
                    elif idx in retrain_neg:            
                        retrain_ben_testset_x.append(ori_testset_x[idx])
                        retrain_ben_testset_y.append(0)
                        
                retrain_mal_testset_x = np.array(retrain_mal_testset_x)        
                retrain_mal_testset_y = np.array(retrain_mal_testset_y)        
                retrain_ben_testset_x = np.array(retrain_ben_testset_x)        
                retrain_ben_testset_y = np.array(retrain_ben_testset_y)        
                        
                # print("retrain_mal_testset_x.shape:",retrain_mal_testset_x.shape)
                # print("retrain_mal_testset_y.shape:",retrain_mal_testset_y.shape)
                # print("retrain_mal_testset_y:",retrain_mal_testset_y)
                """ 
                retrain_mal_testset_x.shape: (3833, 41)
                retrain_mal_testset_y.shape: (3833,)
                """
                
                # print("retrain_ben_testset_x.shape:",retrain_ben_testset_x.shape)
                # print("retrain_ben_testset_y.shape:",retrain_ben_testset_y.shape)    
                # print("retrain_ben_testset_y:",retrain_ben_testset_y)
                
                """ 
                retrain_ben_testset_x.shape: (19, 41)
                retrain_ben_testset_y.shape: (19,)
                """
                
                retrain_testset_x = np.concatenate((retrain_mal_testset_x,retrain_ben_testset_x))
                retrain_testset_y = np.concatenate((retrain_mal_testset_y,retrain_ben_testset_y))
                print("retrain_testset_x.shape:",retrain_testset_x.shape)
                print("retrain_testset_y.shape:",retrain_testset_y.shape)
                # print("retrain_testset_y:",retrain_testset_y)
                        
                """ 
                retrain_testset_x.shape: (3852, 41)
                retrain_testset_y.shape: (3852,)
                """     

                # create trainset
                # retrain_testset_len = len(retrain_testset_x)
                # print("retrain_testset_len:",retrain_testset_len)
                # ori_trainset_x = copy.deepcopy(detector.dataset['train'][0])
                # ori_trainet_y = copy.deepcopy(detector.dataset['train'][1]) 
                                                    
                # retrain_trainset_x = ori_trainset_x[retrain_testset_len*r:retrain_testset_len*(r+1)]
                # retrain_trainset_y = ori_trainet_y[retrain_testset_len*r:retrain_testset_len*(r+1)]
                # print("retrain_trainset_x.shape:",retrain_trainset_x.shape)
                # print("retrain_trainset_y.shape:",retrain_trainset_y.shape)

                # """ 
                # retrain_testset_len: 3852
                # retrain_trainset_x.shape: (3852, 41)
                
                # """
                
                retrain_set_x = np.concatenate((retrain_testset_x,detector.dataset['train'][0]))
                retrain_set_y = np.concatenate((retrain_testset_y,detector.dataset['train'][1]))            
                print("retrain_set_x.shape:",retrain_set_x.shape)
                print("retrain_set_y.shape:",retrain_set_y.shape)            
                # print("retrain_set_y:",retrain_set_y)
                
                """ 
                retrain_set_x.shape: (7704, 41)

                """
                #-----------------------------------------------
                retrainset_x = retrain_set_x
                retrainset_y = retrain_set_y

            #-----------------------------------------------          
            
            print(f">>>>>>>> {r+1} round Retraining {detector.modelname} >>>>>>>>")    
            rou_cost_time=detector.retrain(retrainset_x=retrainset_x, retrainset_y=retrainset_y, timesteps=args.timesteps, curround_exp_result_dir=curround_exp_result_dir)
            
            print(f">>>>>>>> Evaluate {r+1} round retrained {detector.modelname} on clean test data")
            test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1 = detector.test(testset_x=detector.dataset['test'][0], testset_y=detector.dataset['test'][1],timesteps=args.timesteps, exp_result_dir=curround_exp_result_dir)
            
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
    
            test_acc_list.append(test_acc)
            test_los_list.append(test_los)
            test_TP_list.append(test_TP)
            test_FP_list.append(test_FP)
            test_TN_list.append(test_TN)
            test_FN_list.append(test_FN)
            test_recall_list.append(test_recall)
            test_precision_list.append(test_precision)
            test_FPR_list.append(test_FPR)
            test_F1_list.append(test_F1)
            cost_time_list.append(rou_cost_time)

            print(f">>>>>>>> Evaluate {r+1} round retrained {detector.modelname} on adversarial test data")

            # generate adversarial mailicious testset
            print(f"Generate adversarial mailicious exapmples based white-box {detector.modelname}")

            adv_exp_result_dir = os.path.join(curround_exp_result_dir,f'advattack')
            os.makedirs(adv_exp_result_dir, exist_ok=True)
                
            adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps, exp_result_dir=adv_exp_result_dir)
            
            print("adv_testset_x.shape:",adv_testset_x.shape)    
            print("adv_testset_y.shape:",adv_testset_y.shape)    
            """ 
            adv_testset_x.shape: (288,41)
            adv_testset_y.shape: (288,)
            """

            adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_FPR, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps, exp_result_dir=adv_exp_result_dir)
            
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
            
            print(f"{r+1}th-round retrained {detector.modelname} adv_metrics_dic:\n {adv_metrics_dic}")    

            adv_test_acc_list.append(adv_test_acc)
            adv_test_los_list.append(adv_test_los)
            # adv_test_TP_list = []
            # adv_test_FP_list = []
            adv_test_TN_list.append(adv_test_TN)
            adv_test_FN_list.append(adv_test_FN)
            adv_test_recall_list.append(adv_test_recall)
            adv_test_precision_list.append(adv_test_precision)
            # adv_test_FPR_list.append(adv_test_FPR)
            adv_test_F1_list.append(adv_test_F1)
            # adv_cost_time_list.append()
    
    

    
    

        import matplotlib.pyplot as plt
        
        #---------------evaluate on clean testset--------------------------------

        retrain_cle_exp_result_dir = os.path.join(exp_result_dir,f'retrain-evaluate-cle')
        os.makedirs(retrain_cle_exp_result_dir, exist_ok=True)

        loss_png_name = f'Test loss of retrained {detector.modelname}'
        accuracy_png_name = f'Test Accuracy of retrained {detector.modelname}'
        costtime_png_name = f'Cost Time of retrain {detector.modelname}'
        fn_fp_png_name = f'Test FP and FN of retrained {detector.modelname}'
        recall_png_name = f'Test Recall of retrained {detector.modelname}'
        f1_png_name = f'Test F1 of retrained {detector.modelname}'
        
        plt.style.use('seaborn')

        plt.plot(list(range(len(test_los_list))), test_los_list, label='Test Loss', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.xticks(range(min(list(range(len(test_los_list)))), max(list(range(len(test_los_list))))+1, 1))

        plt.legend()
        plt.title(f'{loss_png_name}')
        plt.savefig(f'{retrain_cle_exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(list(range(len(test_acc_list))), test_acc_list, label='Test Accuracy', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.xticks(range(min(list(range(len(test_acc_list)))), max(list(range(len(test_acc_list))))+1, 1))
        plt.legend()
        plt.title(f'{accuracy_png_name}')        
        plt.savefig(f'{retrain_cle_exp_result_dir}/{accuracy_png_name}.png')
        plt.close()

        plt.plot(list(range(len(cost_time_list))), cost_time_list, label='Cost Time', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Cost Time')
        plt.xticks(range(min(list(range(len(cost_time_list)))), max(list(range(len(cost_time_list))))+1, 1))        
        plt.legend()
        plt.title(f'{costtime_png_name}')        
        plt.savefig(f'{retrain_cle_exp_result_dir}/{costtime_png_name}.png')
        plt.close()

        plt.plot(list(range(len(test_FP_list))), test_FP_list, label='Test FP', marker='o')
        plt.plot(list(range(len(test_FN_list))), test_FN_list, label='Test FN', marker='s')
        plt.xlabel('Round')
        plt.ylabel('Test FP and FN')
        plt.xticks(range(min(list(range(len(test_FP_list)))), max(list(range(len(test_FP_list))))+1, 1))        
        plt.legend()
        plt.title(f'{fn_fp_png_name}')        
        plt.savefig(f'{retrain_cle_exp_result_dir}/{fn_fp_png_name}.png')
        plt.close()

        plt.plot(list(range(len(test_recall_list))), test_recall_list, label='Test Recall', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Test Recall')
        plt.xticks(range(min(list(range(len(test_recall_list)))), max(list(range(len(test_recall_list))))+1, 1))        
        plt.legend()
        plt.title(f'{recall_png_name}')        
        plt.savefig(f'{retrain_cle_exp_result_dir}/{recall_png_name}.png')
        plt.close()        

        plt.plot(list(range(len(test_F1_list))), test_F1_list, label='Test F1', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Test F1')
        plt.xticks(range(min(list(range(len(test_F1_list)))), max(list(range(len(test_F1_list))))+1, 1))                
        plt.legend()
        plt.title(f'{f1_png_name}')        
        plt.savefig(f'{retrain_cle_exp_result_dir}/{f1_png_name}.png')
        plt.close()                                 

        #---------------evaluate on adv testset--------------------------------
        retrain_adv_exp_result_dir = os.path.join(exp_result_dir,f'retrain-evaluate-adv')
        os.makedirs(retrain_adv_exp_result_dir, exist_ok=True)

        adv_loss_png_name = f'Test loss of retrained {detector.modelname} on White-box Adv Testset'
        adv_accuracy_png_name = f'Test Accuracy of retrained {detector.modelname} on White-box Adv Testset'
        # adv_costtime_png_name = f'Cost Time of retrain {detector.modelname} on White-box Adv Testset'
        adv_fn_png_name = f'Test FN of retrained {detector.modelname} on White-box Adv Testset'
        adv_recall_png_name = f'Test Recall of retrained {detector.modelname} on White-box Adv Testset'
        adv_f1_png_name = f'Test F1 of retrained {detector.modelname} on White-box Adv Testset'
        
        plt.style.use('seaborn')

        plt.plot(list(range(len(adv_test_los_list))), adv_test_los_list, label='Test Loss', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        # 设置x轴刻度为整数
        plt.xticks(range(min(list(range(len(adv_test_los_list)))), max(list(range(len(adv_test_los_list))))+1, 1))                
        plt.legend()
        plt.title(f'{adv_loss_png_name}')
        plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_loss_png_name}.png')
        plt.close()
                
        plt.plot(list(range(len(adv_test_acc_list))), adv_test_acc_list, label='Test Accuracy', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.xticks(range(min(list(range(len(adv_test_acc_list)))), max(list(range(len(adv_test_acc_list))))+1, 1))                
        plt.legend()
        plt.title(f'{adv_accuracy_png_name}')        
        plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_accuracy_png_name}.png')
        plt.close()

        plt.plot(list(range(len(adv_test_FN_list))), adv_test_FN_list, label='Test False Negative', marker='o')
        plt.xlabel('Round')
        plt.ylabel('FN')
        plt.xticks(range(min(list(range(len(adv_test_FN_list)))), max(list(range(len(adv_test_FN_list))))+1, 1))                
        plt.legend()
        plt.title(f'{adv_fn_png_name}')        
        plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_fn_png_name}.png')
        plt.close()

        plt.plot(list(range(len(adv_test_recall_list))), adv_test_recall_list, label='Test Recall', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Test Recall')
        plt.xticks(range(min(list(range(len(adv_test_recall_list)))), max(list(range(len(adv_test_recall_list))))+1, 1))                
        plt.legend()
        plt.title(f'{adv_recall_png_name}')        
        plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_recall_png_name}.png')
        plt.close()        

        plt.plot(list(range(len(adv_test_F1_list))), adv_test_F1_list, label='Test F1', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Test F1')
        plt.xticks(range(min(list(range(len(adv_test_F1_list)))), max(list(range(len(adv_test_F1_list))))+1, 1))                
        plt.legend()
        plt.title(f'{adv_f1_png_name}')        
        plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_f1_png_name}.png')
        plt.close()                                 
            
            
raise Exception("maggie stop")
