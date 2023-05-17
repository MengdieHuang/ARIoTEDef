""" 
written by Zilin Shen, Daniel de Mello, and Mengdie Huang
Date: May 8, 2023
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from label_packet import generate_label_data, label_packets
from random_perturb import random_perturb
from pso import PSO
from models.lstm import Lstm
from seq2seq.utils import softmax, print_header, get_events
from seq2seq.seq2seq_attention import Seq2seqAttention
# import argparse
import copy
import tensorflow as tf
import random

#--------maggie add packages-------
tf.compat.v1.disable_eager_execution()
from argsparse import get_args
from savedir import set_exp_result_dir
#----------------------------------

#jupyter_args = ['--permute_truncated', '--use_prob_embedding']
args = get_args()
seq2seq_config = {"sequence_length": args.sequence_length, 
                  "permute_truncated": args.permute_truncated,
                  "use_prob_embedding": args.use_prob_embedding,
                  "rv": args.rv
                  }   

#----------maggie add-------------
print("args:",args)
exp_result_dir = set_exp_result_dir(args)
os.makedirs(exp_result_dir, exist_ok=True)
seed = args.seed
#---------------------------------

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# -----------get the preprocessed training and testing saved as .npy files
test_label_infection = np.load('preprocessed/test_label_infection.npy')
train_label_infection = np.load('preprocessed/train_label_infection.npy')
test_data_infection = np.load('preprocessed/test_data_infection.npy')
train_data_infection = np.load('preprocessed/train_data_infection.npy')

test_label_reconnaissance = np.load('preprocessed/test_label_reconnaissance.npy')
train_label_reconnaissance = np.load('preprocessed/train_label_reconnaissance.npy')
test_data_reconnaissance = np.load('preprocessed/test_data_reconnaissance.npy')
train_data_reconnaissance = np.load('preprocessed/train_data_reconnaissance.npy')

test_label_attack = np.load('preprocessed/test_label_attack.npy')
train_label_attack = np.load('preprocessed/train_label_attack.npy')
test_data_attack = np.load('preprocessed/test_data_attack.npy')
train_data_attack = np.load('preprocessed/train_data_attack.npy')

all_data = {"infection": 
                        {
                        'train': [train_data_infection, train_label_infection], 
                        'test': [test_data_infection, test_label_infection]
                        },
            "attack": 
                        {
                        'train': [train_data_attack, train_label_attack], 
                        'test': [test_data_attack, test_label_attack]
                        },
            "reconnaissance": 
                        {
                        'train': [train_data_reconnaissance, train_label_reconnaissance], 
                        'test': [test_data_reconnaissance, test_label_reconnaissance]
                        }
            }

# ----------------create per-step detectors----------------------
ps_attack = Lstm("ps-detector-attack")
ps_attack.add_dataset(all_data['attack']) 

ps_recon = Lstm("ps-detector-recon")
ps_recon.add_dataset(all_data['reconnaissance']) 

ps_infec = Lstm("ps-detector-infec")
ps_infec.add_dataset(all_data['infection']) 

# ----------------train per-step detectors----------------------
print("******************Train Per-Step Detector******************")
metrics_dict = {}
for detector in [ps_attack, ps_recon, ps_infec]:
    #train data
    train_data = detector.dataset['train']
    train_examples = train_data[0]
    train_labels = train_data[1]

    #test data
    test_data = detector.dataset['test']
    test_examples = test_data[0]
    test_labels = test_data[1]
        
    features_len = train_examples.shape[1]
    print('features len is ', features_len)
    
    print_header("Training {} detector".format(detector.name))
    detector.learning(features_len, train_examples, train_labels, kind='', epochs=args.ps_epochs, patience=args.patience)
                    
    print_header("Measureing {} detector performance on test data".format(detector.name))
    _, _, metrics_dict_ps = detector.detection(test_examples, test_labels, kind='')
    print("Metrics: \n", metrics_dict_ps)
    
    """ 
    Metrics: 
    {'auc': 0.5999068250640578, 'accuracy': 0.954169619655091, 'precision': 0.8195876288659794, 'recall': 0.5, 'f1': 0.62109375, 'fp': 35, 'tp': 159, 'fn': 159, 'tn': 3880, 'auprc': 0.39771775272666826, 'precision99': 0.76, 'recall99': 0.0}
    """
    metrics_dict[detector.name] = metrics_dict_ps

#--------maggie add------------
print("******************Adversarial Attack Vanilla Per-Step Detector******************")
for detector in [ps_infec]:
    # print("detector.dataset['test'][0].shape:",detector.dataset['test'][0].shape)
    # print("detector.dataset['test'][1].shape:",detector.dataset['test'][1].shape)
    # print("detector.dataset['test'][1]:",detector.dataset['test'][1])
        
    """ 
    detector.dataset['test'][0].shape: (4233, 41)
    detector.dataset['test'][1].shape: (4233,)
    detector.dataset['test'][1]: [0. 0. 0. ... 0. 0. 0.]
    """
        
    print_header("Measureing {} detector performance on clean test data".format(detector.name))
    _, _, cle_metrics_dict_ps = detector.detection(dataset=detector.dataset['test'][0], label=detector.dataset['test'][1], kind='')
    print("Metrics on clean testset: \n", cle_metrics_dict_ps)
    """ 
    Metrics on clean testset: 
    {'auc': 0.5999068250640578, 'accuracy': 0.954169619655091, 'precision': 0.8195876288659794, 'recall': 0.5, 'f1': 0.62109375, 'fp': 35, 'tp': 159, 'fn': 159, 'tn': 3880, 'auprc': 0.39771775272666826, 'precision99': 0.76, 'recall99': 0.0}
    """
        
    # print_header("Generate adversarial exapmples based white-box {} detector with clean test data".format(detector.name))
    # adv_test_examples, adv_test_labels = detector.advgenerate(cle_testset_x=detector.dataset['test'][0], cle_testset_y=detector.dataset['test'][1], kind='', args=args)

    # generate adversarial infection testset
    print_header("Generate only adversarial mailicious exapmples based white-box {} detector with clean test data".format(detector.name))
    adv_test_examples, adv_test_labels = detector.advgenerate_onlymail(cle_testset_x=detector.dataset['test'][0], cle_testset_y=detector.dataset['test'][1], kind='', args=args)
        
    print("adv_test_examples.shape:",adv_test_examples.shape)
    print("adv_test_labels.shape:",adv_test_labels.shape)
    print("adv_test_labels:",adv_test_labels)
    """ 
    adv_test_examples.shape: (318, 41)
    adv_test_labels.shape: (318,)
    """
    
    # adversarial evasion attack vanilla infection detector
    print_header("Measureing {} detector performance on adversarial test data".format(detector.name))
    _, _, adv_metrics_dict_ps = detector.detection(adv_test_examples, adv_test_labels, kind='')
    print("Metrics on adversarial testset: \n", adv_metrics_dict_ps)
    
    """ 
    Metrics on adversarial testset:
    {'auc': nan, 'accuracy': 0.22641509433962265, 'precision': 1.0, 'recall': 0.22641509433962265, 'f1': 0.36923076923076925, 'fp': 0, 'tp': 72, 'fn': 246, 'tn': 0, 'auprc': 1.0, 'precision99': 0.0, 'recall99': 0.0}
    """
#------------------------------

# metrics_dict_per_round = []
# for r in range(args.relabel_rounds):

#     # -----------------seq2seq stage----------------------
#     #get events
#     events = get_events(ps_attack, ps_recon, ps_infec, all_data['infection']['train'][0])

#     #init seq2seq
#     if args.retrain_seq2seq: 
#         seq2seq = Seq2seqAttention('seq2seq')
#     elif r == 0: 
#         seq2seq = Seq2seqAttention('seq2seq')

#     #train seq2seq
#     seq2seq.learning(events, all_data['infection']['train'][1], seq2seq_config)

#     #get seq2seq tagged events
#     events_preds, tagged_seq2seq = seq2seq.analysis(events, all_data['infection']['train'][1], seq2seq_config)
#     #---------maggie add----------
#     # tagged_seq2seq is the indx list of samples predicted as infection by attention
#     #-----------------------------
    
#     #get per-step infection detector tagged windows
#     preds_ps_infec = ps_infec.predict(all_data['infection']['train'][0], kind='')
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
#     retrain_labels = copy.deepcopy(all_data['infection']['train'][1])
#     retrain_data = copy.deepcopy(all_data['infection']['train'][0])
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

    metrics_dict_per_round = []
    for r in range(args.relabel_rounds):
        print("\n round_index:",r)
        curround_exp_result_dir = os.path.join(exp_result_dir,f'round-{r}')
        
        # -----------------seq2seq stage----------------------
        #get events
        events = get_events(ps_attack, ps_recon, ps_infec, all_data['infection']['train'][0])

        #init seq2seq
        print("args.retrain_seq2seq:",args.retrain_seq2seq)
        # args.retrain_seq2seq: False
                
        if args.retrain_seq2seq: 
            seq2seq = Seq2seqAttention('seq2seq')
        elif r == 0: 
            seq2seq = Seq2seqAttention('seq2seq')

        # else r==1/2/.../r, load pre-trained seq2seq 
                
        # train seq2seq using infection training set
        seq2seq.learning(events, all_data['infection']['train'][1], seq2seq_config)

        #get seq2seq tagged events in training set
        events_preds, tagged_seq2seq = seq2seq.analysis(events, all_data['infection']['train'][1], seq2seq_config)
        #---------maggie add----------
        # tagged_seq2seq is the indx list of samples predicted as infection by attention
        #-----------------------------
        
        #get per-step infection detector tagged windows
        preds_ps_infec = ps_infec.predict(all_data['infection']['train'][0], kind='')
        preds_ps_infec = np.array(preds_ps_infec).squeeze()
        print('preds ps infec shape is', preds_ps_infec.shape)
        tagged_ps_infec = []
        for idx, pred in enumerate(preds_ps_infec):
            if pred>0.5:
                tagged_ps_infec.append(idx)
        #---------maggie add----------
        # tagged_ps_infec is the indx list of samples predicted as infection by per-step infection detector
        #-----------------------------
        
        # # -----------------relabeling---------------------
        # #strategy 1
        # retrain_pos = []
        # retrain_neg = []
        # # for idx in tagged_ps_infec:         
        # #     if idx in tagged_seq2seq:
        # #         retrain_pos.append(idx)     
        # #     else:
        # #         retrain_neg.append(idx)     
        
        # #--------------maggie change---------------    
        # for idx in tagged_ps_infec:         #   all samples in set A
        #     if idx in tagged_seq2seq:       #   all samples in set B
        #         retrain_pos.append(idx)     #   infection array
        #     else:                           #   all samples in set C
        #         retrain_neg.append(idx)     #   benign array    
        # #------------------------------------------
                
        # #overwritte original dataset with new positive and negative labels
        # retrain_labels = copy.deepcopy(all_data['infection']['train'][1])
        # retrain_data = copy.deepcopy(all_data['infection']['train'][0])
        # for idx, l in enumerate(retrain_labels):
        #     if idx in retrain_pos:
        #         retrain_labels[idx] = 1
        #     if idx in retrain_neg:
        #         retrain_labels[idx] = 0
        
        
        
        
        raise Exception("maggie stop here!!!!!!")
        # -----------------seq2seq stage----------------------
        #get events
        events = get_events(ps_attack, ps_recon, ps_infec, all_data['infection']['train'][0])

        #init seq2seq
        if args.retrain_seq2seq: 
            seq2seq = Seq2seqAttention('seq2seq')
        elif r == 0: 
            seq2seq = Seq2seqAttention('seq2seq')

        #train seq2seq
        seq2seq.learning(events, all_data['infection']['train'][1], seq2seq_config)

        #get seq2seq tagged events
        events_preds, tagged_seq2seq = seq2seq.analysis(events, all_data['infection']['train'][1], seq2seq_config)
        #---------maggie add----------
        # tagged_seq2seq is the indx list of samples predicted as infection by attention
        #-----------------------------
        
        #get per-step infection detector tagged windows
        preds_ps_infec = ps_infec.predict(all_data['infection']['train'][0], kind='')
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
        retrain_labels = copy.deepcopy(all_data['infection']['train'][1])
        retrain_data = copy.deepcopy(all_data['infection']['train'][0])
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



    #------------maggie add----------------------
    # for detector in [ps_infec]:
    #     print("detector.dataset['test'][0].shape:",detector.dataset['test'][0].shape)
    #     print("detector.dataset['test'][1].shape:",detector.dataset['test'][1].shape)
    #     print("detector.dataset['test'][1]:",detector.dataset['test'][1])
            
    print_header("Measureing {} detector performance on clean test data".format(detector.name))
    _, _, cle_metrics_dict_ps = detector.detection(dataset=detector.dataset['test'][0], label=detector.dataset['test'][1], kind='')
    print("Metrics on clean testset: \n", cle_metrics_dict_ps)
    """ 
    Metrics on clean testset:
    {'auc': 0.9424592888637991, 'accuracy': 0.9681077250177179, 'precision': 0.9617074701820465, 'recall': 0.9957751056223595, 'f1': 0.978444834743733, 'fp': 122, 'tp': 3064, 'fn': 13, 'tn': 1034, 'auprc': 0.9479619100168108, 'precision99': 0.97, 'recall99': 0.04}
    """
        
    print_header("Generate adversarial exapmples based white-box {} detector with clean test data".format(detector.name))
    adv_test_examples, adv_test_labels = detector.advgenerate(cle_testset_x=detector.dataset['test'][0], cle_testset_y=detector.dataset['test'][1], kind='', args=args)
    print("adv_test_examples.shape:",adv_test_examples.shape)
    print("adv_test_labels.shape:",adv_test_labels.shape)
    
    print_header("Measureing {} detector performance on adversarial test data".format(detector.name))
    _, _, adv_metrics_dict_ps = detector.detection(adv_test_examples, adv_test_labels, kind='')
    print("Metrics on adversarial testset: \n", adv_metrics_dict_ps)
    
    """ 
    Metrics on adversarial testset: 
    {'auc': 0.5382442950432541, 'accuracy': 0.9258209307819514, 'precision': 0.5555555555555556, 'recall': 0.06289308176100629, 'f1': 0.11299435028248589, 'fp': 16, 'tp': 20, 'fn': 298, 'tn': 3899, 'auprc': 0.24193337359270792, 'precision99': 0.01, 'recall99': 0.0}
    """

    #------------------------------







