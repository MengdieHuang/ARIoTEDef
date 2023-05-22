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
import argparse
import copy
import tensorflow as tf
import random
#--------maggie add packages-------
tf.compat.v1.disable_eager_execution()
# from art.attacks.evasion import ProjectedGradientDescent
# from art.estimators.classification import KerasClassifier
#----------------------------------

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)



def get_args(jupyter_args = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--permute_truncated', required=False, action='store_true', 
                        help="Bool for activating permutation invariance")
    parser.add_argument('--retrain_seq2seq', required=False, action='store_true', 
                        help="Bool for retraining seq2seq module from scratch at each relabel round ")
    parser.add_argument('--use_prob_embedding', required=False, action='store_true', 
                        help="Bool for using original probability based embedding proposed in the original paper")
    parser.add_argument('--sequence_length', required=False, type=int, default=10, 
                        help="Length of truncated subsequences used in the seq2seq training")
    parser.add_argument('--rv', required=False, type=int, default=1, 
                        help="'round value' hyperparameter used for probability embedding, if activated")
    parser.add_argument('--ps_epochs', required=False, type=int, default=50, 
                        help="number of training epochs for per-step detectors")
    parser.add_argument('--relabel_rounds', required=False, type=int, default=1, 
                        help="Number of relabel rounds")
    parser.add_argument('--patience', required=False, type=int, default=None,
                        help="Patience for early stopping. Any value activates early stopping.")
    
    #------------maggie add parameters-------
    parser.add_argument('--seed', required=False, type=int, default=0,
                        help="random seed.")  
    parser.add_argument('--eps', required=False, type=float, default=1.0,
                        help="adversarial evasion attack parameter: epsilon.")        
    parser.add_argument('--eps_step', required=False, type=float, default=0.5,
                        help="adversarial evasion attack parameter: epsilon step.")   
    parser.add_argument('--max_iter', required=False, type=int, default=20,
                        help="adversarial evasion attack parameter: iteration number.")           
    #----------------------------------------
    
    if jupyter_args is not None:
        args = parser.parse_args(jupyter_args)
    else: 
        args = parser.parse_args()
    return args

#jupyter_args = ['--permute_truncated', '--use_prob_embedding']
args = get_args()
seq2seq_config = {"sequence_length": args.sequence_length, 
                  "permute_truncated": args.permute_truncated,
                  "use_prob_embedding": args.use_prob_embedding,
                  "rv": args.rv
                  }   
#----------maggie add-------------
print("args:",args)
#---------------------------------

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
    metrics_dict[detector.name] = metrics_dict_ps

#--------maggie add------------
# adversarial evasion attack vanilla infection detector
# generate adversarial infection testset


for detector in [ps_infec]:
    print("detector.dataset['test'][0].shape:",detector.dataset['test'][0].shape)
    print("detector.dataset['test'][1].shape:",detector.dataset['test'][1].shape)
    print("detector.dataset['test'][1]:",detector.dataset['test'][1])
        
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
    {'auc': 0.431708270874543, 'accuracy': 0.27309236947791166, 'precision': nan, 'recall': 0.0, 'f1': nan, 'fp': 0, 'tp': 0, 'fn': 3077, 'tn': 1156, 'auprc': 0.3863027492575339, 'precision99': 0.0, 'recall99': 0.0}
    """

#------------------------------


# raise Exception("maggie stop here!")

metrics_dict_per_round = []
for r in range(args.relabel_rounds):

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

    #get per-step infection detector tagged windows
    preds_ps_infec = ps_infec.predict(all_data['infection']['train'][0], kind='')
    preds_ps_infec = np.array(preds_ps_infec).squeeze()
    print('preds ps infec shape is', preds_ps_infec.shape)
    tagged_ps_infec = []
    for idx, pred in enumerate(preds_ps_infec):
        if pred>0.5:
            tagged_ps_infec.append(idx)


    # -----------------relabeling---------------------
    #strategy 1
    retrain_pos = []
    retrain_neg = []
    for idx in tagged_ps_infec:
        if idx in tagged_seq2seq:
            retrain_pos.append(idx)
        else:
            retrain_neg.append(idx)

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

    #------------maggie add----------------------
    for detector in [ps_infec]:
        print("detector.dataset['test'][0].shape:",detector.dataset['test'][0].shape)
        print("detector.dataset['test'][1].shape:",detector.dataset['test'][1].shape)
        print("detector.dataset['test'][1]:",detector.dataset['test'][1])
            
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
        {'auc': 0.431708270874543, 'accuracy': 0.27309236947791166, 'precision': nan, 'recall': 0.0, 'f1': nan, 'fp': 0, 'tp': 0, 'fn': 3077, 'tn': 1156, 'auprc': 0.3863027492575339, 'precision99': 0.0, 'recall99': 0.0}
        """

    #------------------------------







