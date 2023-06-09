""" 
written by Mengdie Huang, Daniel de Mello, and Zilin Shen
Date: May 8, 2023
"""

import os
os.environ['TF_NUMA_NODES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # set log level WARNING
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
from utils.printformat import print_header
from utils.argsparse import get_args
from utils.savedir import set_exp_result_dir
from utils.events import get_events_from_windows

from data.dataload import loadnpydata
from data.normalize import normalize_multistep_dataset
from models.createmodel import init_psdetector
from models.createmodel import init_seq2seq
import copy
import math
import matplotlib.pyplot as plt

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
    
# -----------normalize sample values    
multistep_dataset = normalize_multistep_dataset(multistep_dataset)

# ----------------create per-step detectors----------------------
reconnaissance_detector, infection_detector, attack_detector = init_psdetector(multistep_dataset, args)

# ----------------train per-step detectors----------------------
print_header("Train Per-Step Detector")
for detector in [reconnaissance_detector, infection_detector, attack_detector]:
    
    if args.stdtrain_pedetector is True:
        
        print(f">>>>>>>> Training {detector.modelname} >>>>>>>>")  
        stdtrain_exp_result_dir = os.path.join(exp_result_dir,f'stdtrain-psdetector')
        os.makedirs(stdtrain_exp_result_dir, exist_ok=True)

        detector.stdtrain(timesteps=args.timesteps, exp_result_dir=stdtrain_exp_result_dir)
        
        print(f">>>>>>>> Evaluate {detector.modelname} on clean test data")
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=detector.dataset['test'][0], testset_y=detector.dataset['test'][1],timesteps=args.timesteps)
        
        metrics_dic = { 
                    'model': detector.modelname,
                    'clean test Accuracy': f'{test_acc*100:.2f}%',
                    'clean test Loss': test_los,
                    'clean test TP': test_TP,
                    'clean test FP': test_FP,
                    'clean test TN': test_TN,
                    'clean test FN': test_FN,
                    'clean test Recall': f'{test_recall*100:.2f}%',
                    'clean test Precision': f'{test_precision*100:.2f}%',
                    'clean test F1': f'{test_F1*100:.2f}%',
                    }
        
        print(f"{detector.modelname} metrics_dic:\n {metrics_dic}")   
        
        detector_save_path = f'{exp_result_dir}/{detector.modelname}-acc-{test_acc:.4f}.h5'
        detector.save_model(detector_save_path)
        
    elif args.stdtrain_pedetector is False:
        print(f">>>>>>>> Evaluate load {detector.modelname} on clean test data")

        pretrain_exp_result_dir = os.path.join(exp_result_dir,f'pretrain-psdetector')
        os.makedirs(pretrain_exp_result_dir, exist_ok=True)
                
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=detector.dataset['test'][0], testset_y=detector.dataset['test'][1],timesteps=args.timesteps)
        
        metrics_dic = { 
                    'model': detector.modelname,
                    'clean test Accuracy': f'{test_acc*100:.2f}%',
                    'clean test Loss': test_los,
                    'clean test TP': test_TP,
                    'clean test FP': test_FP,
                    'clean test TN': test_TN,
                    'clean test FN': test_FN,
                    'clean test Recall': f'{test_recall*100:.2f}%',
                    'clean test Precision': f'{test_precision*100:.2f}%',
                    'clean test F1': f'{test_F1*100:.2f}%',
                    }        
        print(f"{detector.modelname} metrics_dic:\n {metrics_dic}")   
             
# ----------------adversarial attack vanilla per-step detectors----------------------
print_header("Adversarial Attack Vanilla Per-Step Detector")
for detector in [reconnaissance_detector, infection_detector, attack_detector]:
   
    # generate adversarial mailicious testset
    print(f"Generate adversarial mailicious exapmples based white-box {detector.modelname}")

    adv_exp_result_dir = os.path.join(exp_result_dir,f'advattack')
    os.makedirs(adv_exp_result_dir, exist_ok=True)
        
    adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps)
    
    print("adv_testset_x.shape:",adv_testset_x.shape)    
    print("adv_testset_y.shape:",adv_testset_y.shape)    

    adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps)
    
    adv_metrics_dic = { 
                   'model': detector.modelname,
                   'adv test Accuracy': f'{adv_test_acc*100:.2f}%',
                   'adv test Loss': adv_test_los,
                   'adv test TP': adv_test_TP,
                   'adv test FP': adv_test_FP,
                   'adv test TN': adv_test_TN,
                   'adv test FN': adv_test_FN,
                   'adv test Recall': f'{adv_test_recall*100:.2f}%',
                   'adv test Precision': f'{adv_test_precision*100:.2f}%',
                   'adv test F1': f'{adv_test_F1*100:.2f}%',
                }
     
    print(f"{detector.modelname} adv_metrics_dic:\n {adv_metrics_dic}")           
            
# ----------------create seq2seq----------------------
infection_seq2seq = init_seq2seq(multistep_dataset, args)

# ----------------train seq2seq----------------------
print_header("Train Infection Seq2Seq")

for seq2seq in [infection_seq2seq]:
 
    if args.stdtrain_seq2seq is True:
        
        # create trainset
        cle_train_windows_x = seq2seq.dataset['train'][0]
        cle_train_windows_y = seq2seq.dataset['train'][1]
        
        # print("cle_train_windows_x.shape:",cle_train_windows_x.shape)
        # print("cle_train_windows_y.shape:",cle_train_windows_y.shape)
        cle_train_windows_x = cle_train_windows_x.reshape((cle_train_windows_x.shape[0], args.timesteps, int(math.ceil(cle_train_windows_x.shape[1] / args.timesteps))))
        print("cle_train_windows_x.shape:",cle_train_windows_x.shape)
        
        """
        cle_train_windows_x.shape: (19818, 1, 41)
        """
        # 
        # cle_train_windows_x.shape: (19818, 1, 41)
        
        # seq2seq.def_model(input_length=args.sequence_length, output_length =args.sequence_length)

        seq2seq_train_events = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_train_windows_x)
        print("seq2seq_train_events.shape:",seq2seq_train_events.shape)
        """ 
        seq2seq_train_events.shape: (19808, 4)
        """    
        seq2seq.def_model(input_length=args.sequence_length, output_length =args.sequence_length)
        
        stdtrain_exp_result_dir = os.path.join(exp_result_dir,f'stdtrain-seq2seq')
        os.makedirs(stdtrain_exp_result_dir, exist_ok=True)
        
        print(f">>>>>>>> Training {seq2seq.modelname} >>>>>>>>")      
        seq2seq.stdtrain(events=seq2seq_train_events, labels=cle_train_windows_y, exp_result_dir=stdtrain_exp_result_dir)
            
        print(f">>>>>>>> Evaluate {seq2seq.modelname} on clean test data")
        
        # create testset
        cle_test_windows_x = seq2seq.dataset['test'][0]
        cle_test_windows_y = seq2seq.dataset['test'][1]
        # print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
        # print("cle_test_windows_y.shape:",cle_test_windows_y.shape)
        cle_test_windows_x = cle_test_windows_x.reshape((cle_test_windows_x.shape[0], args.timesteps, int(math.ceil(cle_test_windows_x.shape[1] / args.timesteps))))
        print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
        
        """ 
        cle_test_windows_x.shape: (4224, 41)
        cle_test_windows_y.shape: (4224,)
        cle_test_windows_x.shape: (4224, 1, 41)
        """
        
        cle_test_events_x = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_test_windows_x)
        print("cle_test_events_x.shape:",cle_test_events_x.shape)
        """ 
        cle_test_events_x.shape: (4224, 4)
        """        
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = seq2seq.test(events=cle_test_events_x, labels=cle_test_windows_y)
        
        metrics_dic = { 
                    'model': seq2seq.modelname,
                    'clean test Accuracy': f'{test_acc*100:.2f}%',
                    'clean test Loss': test_los,
                    'clean test TP': test_TP,
                    'clean test FP': test_FP,
                    'clean test TN': test_TN,
                    'clean test FN': test_FN,
                    'clean test Recall': f'{test_recall*100:.2f}%',
                    'clean test Precision': f'{test_precision*100:.2f}%',
                    'clean test F1': f'{test_F1*100:.2f}%',
                    }
        
        print(f"{seq2seq.modelname} metrics_dic:\n {metrics_dic}")       

        seq2seq_save_path = f'{exp_result_dir}/{seq2seq.modelname}-acc-{test_acc:.4f}.h5'
        seq2seq.save_model(seq2seq_save_path)

    elif args.stdtrain_seq2seq is False:
        print(f">>>>>>>> Evaluate load {seq2seq.modelname} on clean test data")
        
        pretrain_exp_result_dir = os.path.join(exp_result_dir,f'pretrain-seq2seq')
        os.makedirs(pretrain_exp_result_dir, exist_ok=True)

        
        # create testset
        cle_test_windows_x = seq2seq.dataset['test'][0]
        cle_test_windows_y = seq2seq.dataset['test'][1]
        # print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
        # print("cle_test_windows_y.shape:",cle_test_windows_y.shape)
        cle_test_windows_x = cle_test_windows_x.reshape((cle_test_windows_x.shape[0], args.timesteps, int(math.ceil(cle_test_windows_x.shape[1] / args.timesteps))))
        print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
        
        """ 
        cle_test_windows_x.shape: (4224, 41)
        cle_test_windows_y.shape: (4224,)
        cle_test_windows_x.shape: (4224, 1, 41)
        """
        cle_test_events_x = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_test_windows_x)
        print("cle_test_events_x.shape:",cle_test_events_x.shape)
        """ 
        cle_test_events_x.shape: (4224, 4)
        """        
                
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = seq2seq.test(events=cle_test_events_x, labels=cle_test_windows_y)
        
        metrics_dic = { 
                    'model': seq2seq.modelname,
                    'clean test Accuracy': f'{test_acc*100:.2f}%',
                    'clean test Loss': test_los,
                    'clean test TP': test_TP,
                    'clean test FP': test_FP,
                    'clean test TN': test_TN,
                    'clean test FN': test_FN,
                    'clean test Recall': f'{test_recall*100:.2f}%',
                    'clean test Precision': f'{test_precision*100:.2f}%',
                    'clean test F1': f'{test_F1*100:.2f}%',
                    }
        
        print(f"{seq2seq.modelname} metrics_dic:\n {metrics_dic}")         
                      
# ----------------retrain vanilla per-step detectors----------------------
print_header("Retrain Vanilla Infection Detector")

for detector in [infection_detector]:

    for seq2seq in [infection_seq2seq]:
        
        # vanillia_detector = copy.deepcopy(detector)
        
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
        
        test_FPrate_list=[]
        test_FNrate_list=[]


        adv_test_acc_list = [] 
        adv_test_los_list = []
        adv_test_TN_list = []
        adv_test_FN_list = []
        adv_test_recall_list = []
        adv_test_precision_list = []
        adv_test_F1_list = []
        adv_test_FNrate_list=[]
        
        
        #---------------evaluate vanilia detector---------------
        print(f">>>>>>>> Evaluate vanillia {detector.modelname} on clean test data >>>>>>>>")
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=detector.dataset['test'][0], testset_y=detector.dataset['test'][1],timesteps=args.timesteps)
        
        metrics_dic = { 
                    'model': detector.modelname,
                    'clean test Accuracy': f'{test_acc*100:.2f}%',
                    'clean test Loss': test_los,
                    'clean test TP': test_TP,
                    'clean test FP': test_FP,
                    'clean test TN': test_TN,
                    'clean test FN': test_FN,
                    'clean test Recall': f'{test_recall*100:.2f}%',
                    'clean test Precision': f'{test_precision*100:.2f}%',
                    'clean test F1': f'{test_F1*100:.2f}%',
                    }
        
        print(f"vanillia {detector.modelname} metrics_dic:\n {metrics_dic}")  
        
        FPrate = round((test_FP/(test_FP+test_TN)), 4)
        FNrate = round((test_FN/(test_FN+test_TP)), 4)
        test_FPrate_list.append(FPrate*100)
        test_FNrate_list.append(FNrate*100)
        
        
        test_acc_list.append(test_acc*100)
        test_los_list.append(test_los)
        test_TP_list.append(test_TP)
        test_FP_list.append(test_FP)
        test_TN_list.append(test_TN)
        test_FN_list.append(test_FN)
        test_recall_list.append(test_recall*100)
        test_precision_list.append(test_precision*100)
        test_F1_list.append(test_F1*100)
        cost_time_list.append(0)        
        
                
        print(f">>>>>>>> Evaluate vanillia {detector.modelname} on adversarial test data >>>>>>>>")

        adv_exp_result_dir = os.path.join(exp_result_dir,f'advattack')
        os.makedirs(adv_exp_result_dir, exist_ok=True)
            
        adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps)
        
        print("adv_testset_x.shape:",adv_testset_x.shape)    
        print("adv_testset_y.shape:",adv_testset_y.shape)    
        
        adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps)
        
        adv_metrics_dic = { 
                    'model': detector.modelname,
                    'adv test Accuracy': f'{adv_test_acc*100:.2f}%',
                    'adv test Loss': adv_test_los,
                    'adv test TP': adv_test_TP,
                    'adv test FP': adv_test_FP,
                    'adv test TN': adv_test_TN,
                    'adv test FN': adv_test_FN,
                    'adv test Recall': f'{adv_test_recall*100:.2f}%',
                    'adv test Precision': f'{adv_test_precision*100:.2f}%',
                    'adv test F1': f'{adv_test_F1*100:.2f}%',
                    }
        
        print(f"Vanillia {detector.modelname} adv_metrics_dic:\n {adv_metrics_dic}")    

        adv_FNrate = round((adv_test_FN/(adv_test_FN+adv_test_TP)), 4)
        adv_test_FNrate_list.append(FNrate*100)
        
        adv_test_acc_list.append(adv_test_acc*100)
        adv_test_los_list.append(adv_test_los)
        adv_test_TN_list.append(adv_test_TN)
        adv_test_FN_list.append(adv_test_FN)
        adv_test_recall_list.append(adv_test_recall*100)
        adv_test_precision_list.append(adv_test_precision*100)
        adv_test_F1_list.append(adv_test_F1*100)
        #-----------------------------------------------------    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        for r in range(args.relabel_rounds):
            print(f">>>>>>>>>>>>>> {r+1} Round Retraining: >>>>>>>>>>>>>> ")    
            curround_exp_result_dir = os.path.join(exp_result_dir,f'round-{r+1}')
            os.makedirs(curround_exp_result_dir, exist_ok=True)









            print("args.retrainset_mode:",args.retrainset_mode)
            if args.retrainset_mode == 'advs1':

                print(f">>>>>>>> create {r+1} round retraining dataset >>>>>>>>")    

                #---------------------create event testset----------------   
                # create adv windows testset         
                adv_exp_result_dir = os.path.join(curround_exp_result_dir,f'advattack')
                os.makedirs(adv_exp_result_dir, exist_ok=True)
                    
                adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps)
                
                # print("adv_testset_x.shape:",adv_testset_x.shape)    
                # print("adv_testset_y.shape:",adv_testset_y.shape)    
                """ 
                adv_testset_x.shape: (318, 41)
                adv_testset_y.shape: (318,)
                """                
                
                
                
            
                # create clean windows testset
                cle_test_x = detector.dataset['test'][0]
                cle_test_y = detector.dataset['test'][1]
                # print("cle_test_x.shape:",cle_test_x.shape)
                # print("cle_test_y.shape:",cle_test_y.shape)
                """ 
                cle_test_x.shape: (4233, 41)
                cle_test_y.shape: (4233,)
                """
                
                # create adv+clean windows testset
                test_x = np.concatenate((adv_testset_x,cle_test_x))
                test_y = np.concatenate((adv_testset_y,cle_test_y))            
                # print("test_x.shape:",test_x.shape)
                # print("test_y.shape:",test_y.shape)                 
                """ 
                test_x.shape: (4551, 41)
                test_y.shape: (4551,)
                """

                # create adv+clean events testset
                test_windows_x = copy.deepcopy(test_x)
                test_windows_y = copy.deepcopy(test_y)           
                print("test_windows_x.shape:",test_windows_x.shape)
                print("test_windows_y.shape:",test_windows_y.shape) 
                """ 
                test_windows_x.shape: (4551, 41)
                test_windows_y.shape: (4551,)
                """
                                
                test_windows_x = test_windows_x.reshape((test_windows_x.shape[0], args.timesteps, int(math.ceil(test_windows_x.shape[1] / args.timesteps))))
                
                print("test_windows_x.shape:",test_windows_x.shape)
                
                """ 
                test_windows_x.shape: (4551, 1, 41)
                """
                
                test_events_x = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, test_windows_x)
                print("test_events_x.shape:",test_events_x.shape)
                """ 
                test_events_x.shape: (4551, 4)
                """      




                #---------------------analyze event ----------------   
                # seq2seq predict test_events_x
                seq2seq_tagged_events_probs, seq2seq_tagged_events_idxs = seq2seq.analysis(test_events_x, test_windows_y)
                print('seq2seq_tagged_events_probs.shape:', seq2seq_tagged_events_probs.shape)
                print('seq2seq_tagged_events_idxs.shape:', seq2seq_tagged_events_idxs.shape)
                """ 
                seq2seq_tagged_events_probs.shape: (331,)
                seq2seq_tagged_events_idxs.shape: (331,) 
                """

                #---------------------analyze window ----------------                   
                detector_tagged_windows_probs, detector_tagged_windows_idxs = detector.analysis(test_windows_x)
                print('detector_tagged_windows_probs.shape:', detector_tagged_windows_probs.shape)
                print('detector_tagged_windows_idxs.shape:', detector_tagged_windows_idxs.shape)
                """ 
                detector_tagged_windows_probs.shape: (154,)
                detector_tagged_windows_idxs.shape: (154,)
                """
                
                # implement strategy 1
                set_B_pos_idx = []
                set_C_neg_idx = []

                for idx in seq2seq_tagged_events_idxs:          #   
                    set_B_pos_idx.append(idx)                     #   B
                    
                for idx in detector_tagged_windows_idxs:        #   
                    if idx not in seq2seq_tagged_events_idxs:   #   
                        set_C_neg_idx.append(idx)                 #   C       
                
                
                
                
                
                print(f">>>>>>>> Prepare {r+1} round Retraining clean dataset >>>>>>>>")    

                # extract relabeled test sampels in clean infection test
                # ori_testset_x = test_x
                # ori_testset_y = test_y 
                
                set_B_pos_x=[]
                set_B_pos_y=[]
                set_C_neg_x=[]
                set_C_neg_y=[]   
                
                print("args.strategy:",args.strategy)
                if args.strategy =='strategy1':     
                    for idx, l in enumerate(test_y): # 遍历test set
                        if idx in set_B_pos_idx:                   
                            set_B_pos_x.append(test_x[idx])
                            set_B_pos_y.append(1)
                            
                        elif idx in set_C_neg_idx:            
                            set_C_neg_x.append(test_x[idx])
                            set_C_neg_y.append(0)
                            
                    set_B_pos_x = np.array(set_B_pos_x)        
                    set_B_pos_y = np.array(set_B_pos_y)        
                    set_C_neg_x = np.array(set_C_neg_x)        
                    set_C_neg_y = np.array(set_C_neg_y)        
                            
                    print("set_B_pos_x.shape:",set_B_pos_x.shape)
                    print("set_B_pos_y.shape:",set_B_pos_y.shape)
                    print("set_C_neg_x.shape:",set_C_neg_x.shape)
                    print("set_C_neg_y.shape:",set_C_neg_y.shape)
                                    
                
                    set_BC_x = np.concatenate((set_B_pos_x,set_C_neg_x))
                    set_BC_y = np.concatenate((set_B_pos_y,set_C_neg_y))
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)  
                        
                elif args.strategy =='strategy2': 
                    for idx, l in enumerate(test_y): 
                        if idx in set_B_pos_idx:                   
                            set_B_pos_x.append(test_x[idx])
                            set_B_pos_y.append(1)


                    set_B_pos_x = np.array(set_B_pos_x)        
                    set_B_pos_y = np.array(set_B_pos_y)        
                    print("set_B_pos_x.shape:",set_B_pos_x.shape)
                    print("set_B_pos_y.shape:",set_B_pos_y.shape)
                     
                    set_BC_x = set_B_pos_x
                    set_BC_y = set_B_pos_y
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)  
                    
                                                
                elif args.strategy =='strategy3': 
                    for idx, l in enumerate(test_y):                       
                        if idx in set_C_neg_idx:            
                            set_C_neg_x.append(test_x[idx])
                            set_C_neg_y.append(0)
                                   
                    set_C_neg_x = np.array(set_C_neg_x)        
                    set_C_neg_y = np.array(set_C_neg_y)        
                    print("set_C_neg_x.shape:",set_C_neg_x.shape)
                    print("set_C_neg_y.shape:",set_C_neg_y.shape)
                                
                    set_BC_x = set_C_neg_x
                    set_BC_y = set_C_neg_y
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)              

                elif args.strategy =='strategy4-groundtruth':
                    
                    set_B_pos_x = adv_testset_x 
                    set_B_pos_y = adv_testset_y
                    
                    set_B_pos_x = np.array(set_B_pos_x)        
                    set_B_pos_y = np.array(set_B_pos_y)        
                    print("set_B_pos_x.shape:",set_B_pos_x.shape)
                    print("set_B_pos_y.shape:",set_B_pos_y.shape)
                     
                    set_BC_x = set_B_pos_x
                    set_BC_y = set_B_pos_y
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)  
                     

                retrainset_x = np.concatenate((set_BC_x, detector.dataset['train'][0]))
                retrainset_y = np.concatenate((set_BC_y, detector.dataset['train'][1]))            
                print("retrainset_x.shape:",retrainset_x.shape)
                print("retrainset_y.shape:",retrainset_y.shape)            

            
            print(f">>>>>>>> {r+1} round Retraining {detector.modelname} >>>>>>>>")    
            rou_cost_time=detector.retrain(retrainset_x=retrainset_x, retrainset_y=retrainset_y, timesteps=args.timesteps, curround_exp_result_dir=curround_exp_result_dir)



            #----------------evaluate-------------------------------          
            
            print(f">>>>>>>> Evaluate {r+1} round retrained {detector.modelname} on clean test data")
            test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=detector.dataset['test'][0], testset_y=detector.dataset['test'][1],timesteps=args.timesteps)
            
            metrics_dic = { 
                        'model': detector.modelname,
                        'clean test Accuracy': f'{test_acc*100:.2f}%',
                        'clean test Loss': test_los,
                        'clean test TP': test_TP,
                        'clean test FP': test_FP,
                        'clean test TN': test_TN,
                        'clean test FN': test_FN,
                        'clean test Recall': f'{test_recall*100:.2f}%',
                        'clean test Precision': f'{test_precision*100:.2f}%',
                        'clean test F1': f'{test_F1*100:.2f}%',
                        }
            
            print(f"{detector.modelname} metrics_dic:\n {metrics_dic}")  

            FPrate = round((test_FP/(test_FP+test_TN)), 4)
            FNrate = round((test_FN/(test_FN+test_TP)), 4)
            test_FPrate_list.append(FPrate*100)
            test_FNrate_list.append(FNrate*100)
            
            test_acc_list.append(test_acc*100)
            test_los_list.append(test_los)
            test_TP_list.append(test_TP)
            test_FP_list.append(test_FP)
            test_TN_list.append(test_TN)
            test_FN_list.append(test_FN)
            test_recall_list.append(test_recall*100)
            test_precision_list.append(test_precision*100)
            test_F1_list.append(test_F1*100)
            cost_time_list.append(rou_cost_time)











            print(f">>>>>>>> Evaluate {r+1} round retrained {detector.modelname} on adversarial test data")

            # # generate adversarial mailicious testset
            # print(f"Generate adversarial mailicious exapmples based white-box {detector.modelname}")

            adv_exp_result_dir = os.path.join(curround_exp_result_dir,f'advattack')
            os.makedirs(adv_exp_result_dir, exist_ok=True)
                
            adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps)
            
            # print("adv_testset_x.shape:",adv_testset_x.shape)    
            # print("adv_testset_y.shape:",adv_testset_y.shape)    
            # """ 
            # adv_testset_x.shape: (318, 41)
            # adv_testset_y.shape: (318,)
            # """

            adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps)
            
            adv_metrics_dic = { 
                        'model': detector.modelname,
                        'adv test Accuracy': f'{adv_test_acc*100:.2f}%',
                        'adv test Loss': adv_test_los,
                        'adv test TP': adv_test_TP,
                        'adv test FP': adv_test_FP,
                        'adv test TN': adv_test_TN,
                        'adv test FN': adv_test_FN,
                        'adv test Recall': f'{adv_test_recall*100:.2f}%',
                        'adv test Precision': f'{adv_test_precision*100:.2f}%',
                        'adv test F1': f'{adv_test_F1*100:.2f}%',
                        }
            
            print(f"{r+1}th-round retrained {detector.modelname} adv_metrics_dic:\n {adv_metrics_dic}")    

            adv_FNrate = round((adv_test_FN/(adv_test_FN+adv_test_TP)), 4)
            adv_test_FNrate_list.append(FNrate*100)
        
            adv_test_acc_list.append(adv_test_acc*100)
            adv_test_los_list.append(adv_test_los)
            adv_test_TN_list.append(adv_test_TN)
            adv_test_FN_list.append(adv_test_FN)
            adv_test_recall_list.append(adv_test_recall*100)
            adv_test_precision_list.append(adv_test_precision*100)
            adv_test_F1_list.append(adv_test_F1*100)
        
        #---------------evaluate on clean testset--------------------------------

        retrain_cle_exp_result_dir = os.path.join(exp_result_dir,f'retrain-evaluate-cle')
        os.makedirs(retrain_cle_exp_result_dir, exist_ok=True)

        loss_png_name = f'Test loss of retrained {detector.modelname} on clean testset'
        accuracy_png_name = f'Accuracy of retrained {detector.modelname} on clean testset'
        costtime_png_name = f'Cost Time of retrain {detector.modelname} on clean testset'
        fn_fp_png_name = f'FP and FN of retrained {detector.modelname} on clean testset'
        recall_png_name = f'Recall of retrained {detector.modelname} on clean testset'
        f1_png_name = f'F1 of retrained {detector.modelname} on clean testset'
        fnrate_fprate_png_name = f'FP rate and FN rate of retrained {detector.modelname} on clean testset'
        
        plt.style.use('seaborn')

        plt.plot(list(range(len(test_los_list))), test_los_list, label='Test Loss', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.xticks(range(min(list(range(len(test_los_list)))), max(list(range(len(test_los_list))))+1, 2))

        plt.legend()
        plt.title(f'{loss_png_name}')
        plt.savefig(f'{retrain_cle_exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(list(range(len(test_acc_list))), test_acc_list, label='Test Accuracy', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Accuracy (%)')
        plt.xticks(range(min(list(range(len(test_acc_list)))), max(list(range(len(test_acc_list))))+1, 2))
        plt.legend()
        plt.title(f'{accuracy_png_name}')        
        plt.savefig(f'{retrain_cle_exp_result_dir}/{accuracy_png_name}.png')
        plt.close()

        plt.plot(list(range(len(cost_time_list))), cost_time_list, label='Cost Time', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Cost Time')
        plt.xticks(range(min(list(range(len(cost_time_list)))), max(list(range(len(cost_time_list))))+1, 2))        
        plt.legend()
        plt.title(f'{costtime_png_name}')        
        plt.savefig(f'{retrain_cle_exp_result_dir}/{costtime_png_name}.png')
        plt.close()

        plt.plot(list(range(len(test_FP_list))), test_FP_list, label='Test FP', marker='o')
        plt.plot(list(range(len(test_FN_list))), test_FN_list, label='Test FN', marker='s')
        plt.xlabel('Round')
        plt.ylabel('Test FP and FN')
        plt.xticks(range(min(list(range(len(test_FP_list)))), max(list(range(len(test_FP_list))))+1, 2))        
        plt.legend()
        plt.title(f'{fn_fp_png_name}')        
        plt.savefig(f'{retrain_cle_exp_result_dir}/{fn_fp_png_name}.png')
        plt.close()

        plt.plot(list(range(len(test_FPrate_list))), test_FPrate_list, label='Test FP rate', marker='o')
        plt.plot(list(range(len(test_FNrate_list))), test_FNrate_list, label='Test FN rate', marker='s')
        plt.xlabel('Round')
        plt.ylabel('FP rate and FN rate')
        plt.xticks(range(min(list(range(len(test_FPrate_list)))), max(list(range(len(test_FPrate_list))))+1, 2))        
        plt.legend()
        plt.title(f'{fnrate_fprate_png_name}')        
        plt.savefig(f'{retrain_cle_exp_result_dir}/{fnrate_fprate_png_name}.png')
        plt.close()
        
        
        
        plt.plot(list(range(len(test_recall_list))), test_recall_list, label='Test Recall', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Test Recall (%)')
        plt.xticks(range(min(list(range(len(test_recall_list)))), max(list(range(len(test_recall_list))))+1, 2))        
        plt.legend()
        plt.title(f'{recall_png_name}')        
        plt.savefig(f'{retrain_cle_exp_result_dir}/{recall_png_name}.png')
        plt.close()        

        plt.plot(list(range(len(test_F1_list))), test_F1_list, label='Test F1', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Test F1 (%)')
        plt.xticks(range(min(list(range(len(test_F1_list)))), max(list(range(len(test_F1_list))))+1, 2))                
        plt.legend()
        plt.title(f'{f1_png_name}')        
        plt.savefig(f'{retrain_cle_exp_result_dir}/{f1_png_name}.png')
        plt.close()                                 

        #---------------evaluate on adv testset--------------------------------
        retrain_adv_exp_result_dir = os.path.join(exp_result_dir,f'retrain-evaluate-adv')
        os.makedirs(retrain_adv_exp_result_dir, exist_ok=True)

        adv_loss_png_name = f'Test loss of retrained {detector.modelname} against white-box PGD evasion attack'
        adv_accuracy_png_name = f'Accuracy of retrained {detector.modelname} against white-box PGD evasion attack'
        adv_fn_png_name = f'FN of retrained {detector.modelname} against white-box PGD evasion attack'
        adv_recall_png_name = f'Recall of retrained {detector.modelname} against white-box PGD evasion attack'
        adv_f1_png_name = f'F1 of retrained {detector.modelname} against white-box PGD evasion attack'
        adv_fnrate_png_name = f'FN rate of retrained {detector.modelname} against white-box PGD evasion attack'
        
        plt.style.use('seaborn')

        plt.plot(list(range(len(adv_test_los_list))), adv_test_los_list, label='Test Loss', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        # 设置x轴刻度为整数
        plt.xticks(range(min(list(range(len(adv_test_los_list)))), max(list(range(len(adv_test_los_list))))+1, 2))                
        plt.legend()
        plt.title(f'{adv_loss_png_name}')
        plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_loss_png_name}.png')
        plt.close()
                
        plt.plot(list(range(len(adv_test_acc_list))), adv_test_acc_list, label='Test Accuracy', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Accuracy (%)')
        plt.xticks(range(min(list(range(len(adv_test_acc_list)))), max(list(range(len(adv_test_acc_list))))+1, 2))                
        plt.legend()
        plt.title(f'{adv_accuracy_png_name}')        
        plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_accuracy_png_name}.png')
        plt.close()

        plt.plot(list(range(len(adv_test_FN_list))), adv_test_FN_list, label='Test False Negative', marker='o')
        plt.xlabel('Round')
        plt.ylabel('FN')
        plt.xticks(range(min(list(range(len(adv_test_FN_list)))), max(list(range(len(adv_test_FN_list))))+1, 2))                
        plt.legend()
        plt.title(f'{adv_fn_png_name}')        
        plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_fn_png_name}.png')
        plt.close()

        plt.plot(list(range(len(adv_test_FNrate_list))), adv_test_FNrate_list, label='Test False Negative rate', marker='o')
        plt.xlabel('Round')
        plt.ylabel('FN rate')
        plt.xticks(range(min(list(range(len(adv_test_FNrate_list)))), max(list(range(len(adv_test_FNrate_list))))+1, 2))                
        plt.legend()
        plt.title(f'{adv_fnrate_png_name}')        
        plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_fnrate_png_name}.png')
        plt.close()
        
        
        plt.plot(list(range(len(adv_test_recall_list))), adv_test_recall_list, label='Test Recall', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Test Recall (%)')
        plt.xticks(range(min(list(range(len(adv_test_recall_list)))), max(list(range(len(adv_test_recall_list))))+1, 2))                
        plt.legend()
        plt.title(f'{adv_recall_png_name}')        
        plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_recall_png_name}.png')
        plt.close()        

        plt.plot(list(range(len(adv_test_F1_list))), adv_test_F1_list, label='Test F1', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Test F1 (%)')
        plt.xticks(range(min(list(range(len(adv_test_F1_list)))), max(list(range(len(adv_test_F1_list))))+1, 2))                
        plt.legend()
        plt.title(f'{adv_f1_png_name}')        
        plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_f1_png_name}.png')
        plt.close()                                 
            
            
raise Exception("maggie stop")
