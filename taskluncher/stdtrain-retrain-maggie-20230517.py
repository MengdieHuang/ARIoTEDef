""" 
written by Zilin Shen, Daniel de Mello, and Mengdie Huang
Date: May 8, 2023
"""

import os
os.environ['TF_NUMA_NODES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置日志级别为 WARNING

import numpy as np
# from models.lstm import Lstm
from seq2seq.utils import print_header, get_events
from seq2seq.seq2seq_attention import Seq2seqAttention
import copy
import tensorflow as tf
# import random

tf.compat.v1.disable_eager_execution()
from argsparse import get_args
from savedir import set_exp_result_dir
from data.dataload import loadnpydata
# from models.keraslstm import PSDetector
from models.createmodel import init_psdetector

# -----------parse parameters-----------
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

# -----------get the preprocessed training and testing saved as .npy files
multistep_dataset = loadnpydata()

# ----------------create per-step detectors----------------------
attack_detector, reconnaissance_detector, infection_detector = init_psdetector(multistep_dataset, args)


# ----------------train per-step detectors----------------------
print_header("Train Per-Step Detector")
for detector in [attack_detector, reconnaissance_detector, infection_detector]:
    print(f">>>>>>>> Training {detector.modelname} >>>>>>>>")    

    detector.stdtrain()

raise Exception("maggie stop here")

# metrics_dict = {}
for detector in [ps_attack, ps_recon, ps_infec]:
    print_header("Training {} detector".format(detector.name))    
    #train data
    # train_data = detector.dataset['train']
    # train_examples = train_data[0]
    # train_labels = train_data[1]
    # cle_trainset_x = detector.dataset['train'][0]
    # cle_trainset_y = detector.dataset['train'][1]
    
    detector.train(args)
    
        
    # features_len = train_examples.shape[1]
    # print('features len is ', features_len)
    
    # print_header("Training {} detector".format(detector.name))
    detector.learning(features_len, train_examples, train_labels, kind='', epochs=args.ps_epochs, patience=args.patience)
                    
    print_header("Measureing {} detector performance on test data".format(detector.name))
    #test data
    test_data = detector.dataset['test']
    test_examples = test_data[0]
    test_labels = test_data[1]
    
    _, _, metrics_dict_ps = detector.detection(test_examples, test_labels, kind='')
    print("Metrics: \n", metrics_dict_ps)
    
    # metrics_dict[detector.name] = metrics_dict_ps

# #--------maggie add------------
# print("******************Adversarial Attack Vanilla Per-Step Detector******************")
# for detector in [ps_infec]:
#     # print("detector.dataset['test'][0].shape:",detector.dataset['test'][0].shape)
#     # print("detector.dataset['test'][1].shape:",detector.dataset['test'][1].shape)
#     # print("detector.dataset['test'][1]:",detector.dataset['test'][1])
        
#     """ 
#     detector.dataset['test'][0].shape: (4233, 41)
#     detector.dataset['test'][1].shape: (4233,)
#     detector.dataset['test'][1]: [0. 0. 0. ... 0. 0. 0.]
#     """
        
#     print_header("Measureing {} detector performance on clean test data".format(detector.name))
#     _, _, cle_metrics_dict_ps = detector.detection(dataset=detector.dataset['test'][0], label=detector.dataset['test'][1], kind='')
#     print("Metrics on clean testset: \n", cle_metrics_dict_ps)
#     """ 
#     Metrics on clean testset: 
#     {'auc': 0.5999068250640578, 'accuracy': 0.954169619655091, 'precision': 0.8195876288659794, 'recall': 0.5, 'f1': 0.62109375, 'fp': 35, 'tp': 159, 'fn': 159, 'tn': 3880, 'auprc': 0.39771775272666826, 'precision99': 0.76, 'recall99': 0.0}
#     """
        
#     # print_header("Generate adversarial exapmples based white-box {} detector with clean test data".format(detector.name))
#     # adv_test_examples, adv_test_labels = detector.advgenerate(cle_testset_x=detector.dataset['test'][0], cle_testset_y=detector.dataset['test'][1], kind='', args=args)

#     # generate adversarial infection testset
#     print_header("Generate only adversarial mailicious exapmples based white-box {} detector with clean test data".format(detector.name))
#     adv_test_examples, adv_test_labels = detector.advgenerate_onlymail(cle_testset_x=detector.dataset['test'][0], cle_testset_y=detector.dataset['test'][1], kind='', args=args)
        
#     print("adv_test_examples.shape:",adv_test_examples.shape)
#     print("adv_test_labels.shape:",adv_test_labels.shape)
#     print("adv_test_labels:",adv_test_labels)
#     """ 
#     adv_test_examples.shape: (318, 41)
#     adv_test_labels.shape: (318,)
#     """
    
#     # adversarial evasion attack vanilla infection detector
#     print_header("Measureing {} detector performance on adversarial test data".format(detector.name))
#     _, _, adv_metrics_dict_ps = detector.detection(adv_test_examples, adv_test_labels, kind='')
#     print("Metrics on adversarial testset: \n", adv_metrics_dict_ps)
    
#     """ 
#     Metrics on adversarial testset:
#     {'auc': nan, 'accuracy': 0.22641509433962265, 'precision': 1.0, 'recall': 0.22641509433962265, 'f1': 0.36923076923076925, 'fp': 0, 'tp': 72, 'fn': 246, 'tn': 0, 'auprc': 1.0, 'precision99': 0.0, 'recall99': 0.0}
#     """
# #------------------------------

# metrics_dict_per_round = []
# for r in range(args.relabel_rounds):

#     # -----------------seq2seq stage----------------------
#     #get events
#     events = get_events(ps_attack, ps_recon, ps_infec, multistep_dataset['infection']['train'][0])

#     #init seq2seq
#     if args.retrain_seq2seq: 
#         seq2seq = Seq2seqAttention('seq2seq')
#     elif r == 0: 
#         seq2seq = Seq2seqAttention('seq2seq')

#     #train seq2seq
#     seq2seq.learning(events, multistep_dataset['infection']['train'][1], seq2seq_config)

#     #get seq2seq tagged events
#     events_preds, tagged_seq2seq = seq2seq.analysis(events, multistep_dataset['infection']['train'][1], seq2seq_config)
#     #---------maggie add----------
#     # tagged_seq2seq is the indx list of samples predicted as infection by attention
#     #-----------------------------
    
#     #get per-step infection detector tagged windows
#     preds_ps_infec = ps_infec.predict(multistep_dataset['infection']['train'][0], kind='')
#     preds_ps_infec = np.array(preds_ps_infec).squeeze()
#     print('preds ps infec shape is', preds_ps_infec.shape)
#     tagged_ps_infec = []
#     for idx, pred in enumerate(preds_ps_infec):
#         if pred>0.5:
#             tagged_ps_infec.append(idx)
#     #---------maggie add----------
#     # tagged_ps_infec is the indx list of samples predicted as infection by per-step infection detector
#     #-----------------------------
    
#     # -----------------relabeling---------------------
#     #strategy 1
#     retrain_pos = []
#     retrain_neg = []
#     # for idx in tagged_ps_infec:         
#     #     if idx in tagged_seq2seq:
#     #         retrain_pos.append(idx)     
#     #     else:
#     #         retrain_neg.append(idx)     
    
#     #--------------maggie change---------------    
#     for idx in tagged_ps_infec:         #   all samples in set A
#         if idx in tagged_seq2seq:       #   all samples in set B
#             retrain_pos.append(idx)     #   infection array
#         else:                           #   all samples in set C
#             retrain_neg.append(idx)     #   benign array    
#     #------------------------------------------
            
#     #overwritte original dataset with new positive and negative labels
#     retrain_labels = copy.deepcopy(multistep_dataset['infection']['train'][1])
#     retrain_data = copy.deepcopy(multistep_dataset['infection']['train'][0])
#     for idx, l in enumerate(retrain_labels):
#         if idx in retrain_pos:
#             retrain_labels[idx] = 1
#         if idx in retrain_neg:
#             retrain_labels[idx] = 0
            
#     # -----------------retrain per-step infection detector with new labels---------------------
#     #def retrain_detector(detector, retrain_data, retrain_labels, test_data, test_labels):

#     features_len = retrain_data.shape[1]
#     print('features len is ', features_len)

#     print_header("Retraining {} detector".format('infection'))
#     ps_infec.learning(features_len, retrain_data, retrain_labels, kind='', epochs=args.ps_epochs)
                    
#     print_header("Measureing {} detector performance on test data".format('infection'))
#     _, _, metrics_dict_new = ps_infec.detection(detector.dataset['test'][0], detector.dataset['test'][1], kind='')

#     print_header("Per-step infection detector metrics BEFORE relabeling")
#     print( metrics_dict[ps_infec.name])

#     print_header("Per-step infection detector metrics AFTER relabeling")
#     metrics_dict_per_round.append(metrics_dict_new)
#     for r, mdround in enumerate(metrics_dict_per_round):
#         print(f'Round {r}:')
#         print(mdround)


#-----------maggie add---------------
print("******************Retrain Vanilla Per-Step Detector******************")
for detector in [ps_infec]: # only retrain infection detector
    print_header("Retrain {} detector".format(detector.name))
    cle_testset_x = detector.dataset['test'][0]
    cle_testset_y = detector.dataset['test'][1]
    cle_trainset_x = detector.dataset['train'][0]
    cle_trainset_y = detector.dataset['train'][1]
    print("cle_testset_x.shape:",cle_testset_x.shape)
    print("cle_testset_y.shape:",cle_testset_y.shape)
    print("cle_trainset_x.shape:",cle_trainset_x.shape)
    print("cle_trainset_y.shape:",cle_trainset_y.shape)            
    
    """
    cle_testset_x.shape: (4233, 41)
    cle_testset_y.shape: (4233,)
    cle_trainset_x.shape: (19818, 41)
    cle_trainset_y.shape: (19818,)
    """  

    # init seq2seq
    seq2seq = Seq2seqAttention('seq2seq')              

    for r in range(args.relabel_rounds):
        print("\n round_index:",r)
        curround_exp_result_dir = os.path.join(exp_result_dir,f'round-{r}')
         
        print_header("{}-th train seq2seq".format(r))
        #get events
        train_events = get_events(ps_attack, ps_recon, ps_infec, cle_trainset_x)                
        # train seq2seq using events from infection training set
        seq2seq.learning(train_events, cle_trainset_y, seq2seq_config)
        
        
        # on clean exampels
        # -----------------seq2seq stage----------------------        
        print_header("{}-th trained seq2seq predict test events".format(r))
        test_events = get_events(ps_attack, ps_recon, ps_infec, cle_testset_x) 

        #get seq2seq tagged events in training set
        test_events_preds, tagged_seq2seq = seq2seq.analysis(test_events, cle_testset_y, seq2seq_config)
        """ 
        test_events_preds is the probality list of samples predicted as infection by seq2seq
        tagged_seq2seq is the indx list of samples in test set predicted as infection by seq2seq        
        """

        # -----------------per-step detector stage----------------------
        print_header("{}-th trained per-step detector predict test windows".format(r))        
        # get per-step infection detector tagged windows
        preds_detector = detector.predict(cle_testset_x, kind='')
        preds_detector = np.array(preds_detector).squeeze()
        # print('preds_detector.shape:', preds_detector.shape)
        # preds ps infec shape is (19818,)
        
        # dataset A
        tagged_detector = []
        for idx, pred in enumerate(preds_detector):
            if pred>0.5:    # predicted label = infection
                tagged_detector.append(idx)
        """ 
        tagged_detector is the indx list of samples in test set predicted as infection by infection detector
        """       
        # -----------------relabeling---------------------
        #strategy 1
        # dataset A tagged_detector
        
        # dataset B tagged_seq2seq
        
        # dataset C A/B
        
        retrain_pos = []
        retrain_neg = []

        for idx in tagged_detector:         #   all samples in set A
            if idx in tagged_seq2seq:       #   all samples in set B
                retrain_pos.append(idx)     #   infection array
            else:                           #   all samples in set C
                retrain_neg.append(idx)     #   benign array    


        # extrac relabeled test sampels in clean infection test
        ori_cle_testset_x = copy.deepcopy(cle_testset_x)
        ori_cle_testset_y = copy.deepcopy(cle_testset_y)        
        
        retrain_cle_mal_testset_x=[]
        retrain_cle_mal_testset_y=[]
        retrain_cle_ben_testset_x=[]
        retrain_cle_ben_testset_y=[]        
        for idx, l in enumerate(ori_cle_testset_y): # 遍历test set
            if idx in retrain_pos:      # infection 样本                
                retrain_cle_mal_testset_x.append(ori_cle_testset_x[idx])
                retrain_cle_mal_testset_y.append(1)
                # retrain_cle_mal_testset_y.append(ori_cle_testset_y[idx])
                
            elif idx in retrain_neg:    # benign 样本
                retrain_cle_ben_testset_x.append(ori_cle_testset_x[idx])
                retrain_cle_ben_testset_y.append(0)
                # retrain_cle_ben_testset_y.append(ori_cle_testset_y[idx])
                
        retrain_cle_mal_testset_x = np.array(retrain_cle_mal_testset_x)        
        retrain_cle_mal_testset_y = np.array(retrain_cle_mal_testset_y)        
        retrain_cle_ben_testset_x = np.array(retrain_cle_ben_testset_x)        
        retrain_cle_ben_testset_y = np.array(retrain_cle_ben_testset_y)        
                
        print("retrain_cle_mal_testset_x.shape:",retrain_cle_mal_testset_x.shape)
        print("retrain_cle_mal_testset_y.shape:",retrain_cle_mal_testset_y.shape)
        print("retrain_cle_mal_testset_y:",retrain_cle_mal_testset_y)
        print("retrain_cle_ben_testset_x.shape:",retrain_cle_ben_testset_x.shape)
        print("retrain_cle_ben_testset_y.shape:",retrain_cle_ben_testset_y.shape)    
        print("retrain_cle_ben_testset_y:",retrain_cle_ben_testset_y)
        
        retrain_cle_testset_x = np.concatenate((retrain_cle_mal_testset_x,retrain_cle_ben_testset_x))
        retrain_cle_testset_y = np.concatenate((retrain_cle_mal_testset_y,retrain_cle_ben_testset_y))
        print("retrain_cle_testset_x.shape:",retrain_cle_testset_x.shape)
        print("retrain_cle_testset_y.shape:",retrain_cle_testset_y.shape)
        print("retrain_cle_testset_y:",retrain_cle_testset_y)
                
        """ 
        retrain_cle_mal_testset_x.shape: (380, 41)
        retrain_cle_mal_testset_y.shape: (380,)
        retrain_cle_ben_testset_x.shape: (19, 41)
        retrain_cle_ben_testset_y.shape: (19,)
        retrain_cle_testset_x.shape: (399, 41)
        retrain_cle_testset_y.shape: (399,)
        """                
                
                
                
                
                
        # on adversarial exampels
        print_header("Measureing {} detector performance on clean test data".format(detector.name))
        _, _, cle_metrics_dict_ps = detector.detection(dataset=cle_testset_x, label=cle_testset_y, kind='')
        print("Metrics on clean testset: \n", cle_metrics_dict_ps)

        # generate adversarial infection testset
        print_header("Generate only adversarial mailicious exapmples based white-box {} detector with clean test data".format(detector.name))
        adv_test_examples, adv_test_labels = detector.advgenerate_onlymail(cle_testset_x=cle_testset_x, cle_testset_y=cle_testset_y, kind='', args=args)
            
        print("adv_test_examples.shape:",adv_test_examples.shape)
        print("adv_test_labels.shape:",adv_test_labels.shape)
        # print("adv_test_labels:",adv_test_labels)
        """ 
        adv_test_examples.shape: (318, 41)
        adv_test_labels.shape: (318,)
        """
        
        # adversarial evasion attack vanilla infection detector
        print_header("Measureing {} detector performance on adversarial test data".format(detector.name))
        _, _, adv_metrics_dict_ps = detector.detection(adv_test_examples, adv_test_labels, kind='')
        print("Metrics on adversarial testset: \n", adv_metrics_dict_ps)
        
            
            
        adv_testset_x = adv_test_examples
        adv_testset_y = adv_test_labels
        
        # -----------------seq2seq stage----------------------        
        print_header("{}-th trained seq2seq predict adversarial malicious test events".format(r))
        test_events = get_events(ps_attack, ps_recon, ps_infec, adv_testset_x) 

        #get seq2seq tagged events in training set
        test_events_preds, tagged_seq2seq = seq2seq.analysis(test_events, adv_testset_y, seq2seq_config)
        """ 
        test_events_preds is the probality list of samples predicted as infection by seq2seq
        tagged_seq2seq is the indx list of samples in test set predicted as infection by seq2seq        
        """

        # -----------------per-step detector stage----------------------
        print_header("{}-th trained per-step detector predict adversarial malicious test windows".format(r))        
        # get per-step infection detector tagged windows
        preds_detector = detector.predict(adv_testset_x, kind='')
        preds_detector = np.array(preds_detector).squeeze()
        print('preds_detector.shape:', preds_detector.shape)
        # preds_detector.shape: (318,)
        
        # dataset A
        tagged_detector = []
        for idx, pred in enumerate(preds_detector):
            if pred>0.5:    # predicted label = infection
                tagged_detector.append(idx)
        """ 
        tagged_detector is the indx list of samples in test set predicted as infection by infection detector
        """       
        # -----------------relabeling---------------------
        #strategy 1
        # dataset A tagged_detector
        
        # dataset B tagged_seq2seq
        
        # dataset C A/B
        
        retrain_pos = []
        retrain_neg = []

        for idx in tagged_detector:         #   all samples in set A
            if idx in tagged_seq2seq:       #   all samples in set B
                retrain_pos.append(idx)     #   infection array
            else:                           #   all samples in set C
                retrain_neg.append(idx)     #   benign array    
                                

        # extrac relabeled test sampels in adversarial infection test
        ori_adv_testset_x = copy.deepcopy(adv_testset_x)
        ori_adv_testset_y = copy.deepcopy(adv_testset_y)        
        
        retrain_adv_mal_testset_x=[]
        retrain_adv_mal_testset_y=[]
        retrain_adv_ben_testset_x=[]
        retrain_adv_ben_testset_y=[]        
        for idx, l in enumerate(ori_adv_testset_y): # 遍历test set
            if idx in retrain_pos:      # infection 样本                
                retrain_adv_mal_testset_x.append(ori_adv_testset_x[idx])
                retrain_adv_mal_testset_y.append(ori_adv_testset_y[idx])
                
            elif idx in retrain_neg:    # benign 样本
                retrain_adv_ben_testset_x.append(ori_adv_testset_x[idx])
                retrain_adv_ben_testset_y.append(ori_adv_testset_y[idx])
                
        retrain_adv_mal_testset_x = np.array(retrain_adv_mal_testset_x)        
        retrain_adv_mal_testset_y = np.array(retrain_adv_mal_testset_y)        
        retrain_adv_ben_testset_x = np.array(retrain_adv_ben_testset_x)        
        retrain_adv_ben_testset_y = np.array(retrain_adv_ben_testset_y)        
                
        print("retrain_adv_mal_testset_x.shape:",retrain_adv_mal_testset_x.shape)
        print("retrain_adv_mal_testset_y.shape:",retrain_adv_mal_testset_y.shape)
        print("retrain_adv_ben_testset_x.shape:",retrain_adv_ben_testset_x.shape)
        print("retrain_adv_ben_testset_y.shape:",retrain_adv_ben_testset_y.shape)
        
        
        retrain_testset_x = np.concatenate((retrain_cle_testset_x,retrain_adv_mal_testset_x))
        retrain_testset_y = np.concatenate((retrain_cle_testset_y,retrain_adv_mal_testset_y))
        print("retrain_testset_x.shape:",retrain_testset_x.shape)
        print("retrain_testset_y.shape:",retrain_testset_y.shape)
        print("retrain_testset_y:",retrain_testset_y)
                
                        
        raise Exception("maggie stop here!!!!!!")
        # -----------------seq2seq stage----------------------
        #get events
        events = get_events(ps_attack, ps_recon, ps_infec, multistep_dataset['infection']['train'][0])

        #init seq2seq
        if args.retrain_seq2seq: 
            seq2seq = Seq2seqAttention('seq2seq')
        elif r == 0: 
            seq2seq = Seq2seqAttention('seq2seq')

        #train seq2seq
        seq2seq.learning(events, multistep_dataset['infection']['train'][1], seq2seq_config)

        #get seq2seq tagged events
        events_preds, tagged_seq2seq = seq2seq.analysis(events, multistep_dataset['infection']['train'][1], seq2seq_config)
        #---------maggie add----------
        # tagged_seq2seq is the indx list of samples predicted as infection by attention
        #-----------------------------
        
        #get per-step infection detector tagged windows
        preds_ps_infec = ps_infec.predict(multistep_dataset['infection']['train'][0], kind='')
        preds_ps_infec = np.array(preds_ps_infec).squeeze()
        print('preds ps infec shape is', preds_ps_infec.shape)
        tagged_ps_infec = []
        for idx, pred in enumerate(preds_ps_infec):
            if pred>0.5:
                tagged_ps_infec.append(idx)
        #---------maggie add----------
        # tagged_ps_infec is the indx list of samples predicted as infection by per-step infection detector
        #-----------------------------
        
        # -----------------relabeling---------------------
        #strategy 1
        retrain_pos = []
        retrain_neg = []
        # for idx in tagged_ps_infec:         
        #     if idx in tagged_seq2seq:
        #         retrain_pos.append(idx)     
        #     else:
        #         retrain_neg.append(idx)     
        
        #--------------maggie change---------------    
        for idx in tagged_ps_infec:         #   all samples in set A
            if idx in tagged_seq2seq:       #   all samples in set B
                retrain_pos.append(idx)     #   infection array
            else:                           #   all samples in set C
                retrain_neg.append(idx)     #   benign array    
        #------------------------------------------
                
        #overwritte original dataset with new positive and negative labels
        retrain_labels = copy.deepcopy(multistep_dataset['infection']['train'][1])
        retrain_data = copy.deepcopy(multistep_dataset['infection']['train'][0])
        for idx, l in enumerate(retrain_labels):
            if idx in retrain_pos:
                retrain_labels[idx] = 1
            if idx in retrain_neg:
                retrain_labels[idx] = 0
                
        # -----------------retrain per-step infection detector with new labels---------------------
        #def retrain_detector(detector, retrain_data, retrain_labels, test_data, test_labels):

        features_len = retrain_data.shape[1]
        print('features len is ', features_len)

        print_header("Retraining {} detector".format('infection'))
        ps_infec.learning(features_len, retrain_data, retrain_labels, kind='', epochs=args.ps_epochs)
                        
        print_header("Measureing {} detector performance on test data".format('infection'))
        _, _, metrics_dict_new = ps_infec.detection(detector.dataset['test'][0], detector.dataset['test'][1], kind='')

        print_header("Per-step infection detector metrics BEFORE relabeling")
        print( metrics_dict[ps_infec.name])

        print_header("Per-step infection detector metrics AFTER relabeling")
        metrics_dict_per_round.append(metrics_dict_new)
        for r, mdround in enumerate(metrics_dict_per_round):
            print(f'Round {r}:')
            print(mdround)

#------------------------------------



    # #------------maggie add----------------------
    # # for detector in [ps_infec]:
    # #     print("detector.dataset['test'][0].shape:",detector.dataset['test'][0].shape)
    # #     print("detector.dataset['test'][1].shape:",detector.dataset['test'][1].shape)
    # #     print("detector.dataset['test'][1]:",detector.dataset['test'][1])
            
    # print_header("Measureing {} detector performance on clean test data".format(detector.name))
    # _, _, cle_metrics_dict_ps = detector.detection(dataset=detector.dataset['test'][0], label=detector.dataset['test'][1], kind='')
    # print("Metrics on clean testset: \n", cle_metrics_dict_ps)
    # """ 
    # Metrics on clean testset:
    # {'auc': 0.9424592888637991, 'accuracy': 0.9681077250177179, 'precision': 0.9617074701820465, 'recall': 0.9957751056223595, 'f1': 0.978444834743733, 'fp': 122, 'tp': 3064, 'fn': 13, 'tn': 1034, 'auprc': 0.9479619100168108, 'precision99': 0.97, 'recall99': 0.04}
    # """
        
    # print_header("Generate adversarial exapmples based white-box {} detector with clean test data".format(detector.name))
    # adv_test_examples, adv_test_labels = detector.advgenerate(cle_testset_x=detector.dataset['test'][0], cle_testset_y=detector.dataset['test'][1], kind='', args=args)
    # print("adv_test_examples.shape:",adv_test_examples.shape)
    # print("adv_test_labels.shape:",adv_test_labels.shape)
    
    # print_header("Measureing {} detector performance on adversarial test data".format(detector.name))
    # _, _, adv_metrics_dict_ps = detector.detection(adv_test_examples, adv_test_labels, kind='')
    # print("Metrics on adversarial testset: \n", adv_metrics_dict_ps)
    
    # """ 
    # Metrics on adversarial testset: 
    # {'auc': 0.5382442950432541, 'accuracy': 0.9258209307819514, 'precision': 0.5555555555555556, 'recall': 0.06289308176100629, 'f1': 0.11299435028248589, 'fp': 16, 'tp': 20, 'fn': 298, 'tn': 3899, 'auprc': 0.24193337359270792, 'precision99': 0.01, 'recall99': 0.0}
    # """

    # #------------------------------







