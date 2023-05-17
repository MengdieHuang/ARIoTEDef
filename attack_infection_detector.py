#written by Maggie Huang
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from label_packet import generate_label_data, label_packets
from random_perturb import random_perturb
# from pso import PSO
from models.lstm import Lstm
from seq2seq.utils import softmax, print_header, get_events
from seq2seq.seq2seq_attention import Seq2seqAttention
import argparse
import copy
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import random

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

print("train_data_infection.shape:",train_data_infection.shape)
print("train_label_infection.shape:",train_label_infection.shape)
# print("train_data_infection:",train_data_infection)
print("test_data_infection.shape:",test_data_infection.shape)
print("test_label_infection.shape:",test_label_infection.shape)
# print("test_data_infection:",test_data_infection)


"""
train_data_infection.shape: (19818, 41)
train_label_infection.shape: (19818,)
test_data_infection.shape: (4233, 41)
test_label_infection.shape: (4233,)
train_data_infection: [[ 9.71921e-01  4.40000e-01  3.31000e-04 ...  5.50000e+02  6.86400e+03
   7.04840e+05]
 [-1.00000e+00 -1.00000e+00  9.99990e+04 ...  1.00000e+00  0.00000e+00
   1.45800e+03]
 [-1.00000e+00 -1.00000e+00  9.99990e+04 ...  7.40000e+01  0.00000e+00
   1.00182e+05]
 ...
 [-1.00000e+00 -1.00000e+00  9.99990e+04 ...  4.00000e+00  0.00000e+00
   2.28000e+02]
 [-1.00000e+00 -1.00000e+00  9.99990e+04 ...  7.00000e+00  0.00000e+00
   4.03000e+02]
 [-1.00000e+00 -1.00000e+00  9.99990e+04 ...  1.00000e+00  0.00000e+00
   5.80000e+01]]
"""

from keras.models import load_model

infection_classifier = load_model('/home/huan1932/IoTDataGenerate/data_preprocess/result/model/vanilla_ps-detector-attack-_detector-v1.h5')

print("finished loading vanilla infection detector")
print("evaluate the performance of the vanilla infection detector on clean test")

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier


# 示例数据

# Min-Max Scaling
test_data_infection_min = np.min(test_data_infection)
print("test_data_infection_min:",test_data_infection_min)
test_data_infection_max = np.max(test_data_infection)
print("test_data_infection_max:",test_data_infection_max)
"""
test_data_infection_min: -1.0
test_data_infection_max: 920098.0
"""

normalized_test_data_infection = (test_data_infection - test_data_infection_min) / (test_data_infection_max - test_data_infection_min)

print("normalized_test_data_infection.shape:",normalized_test_data_infection.shape)
print("test_label_infection.shape:",test_label_infection.shape)
print("normalized_test_data_infection:",normalized_test_data_infection)
"""
normalized_test_data_infection: [[2.14007840e-06 1.65199614e-06 1.08764274e-06 ... 2.06499518e-04
  7.89154211e-03 2.37555959e-01]
 [0.00000000e+00 0.00000000e+00 1.08683957e-01 ... 3.26051871e-06
  1.08683957e-06 1.27160229e-04]
 [0.00000000e+00 0.00000000e+00 1.08683957e-01 ... 3.26051871e-06
  1.08683957e-06 1.27160229e-04]
 ...
 [0.00000000e+00 0.00000000e+00 1.08683957e-01 ... 1.05423438e-04
  1.08683957e-06 7.72199513e-03]
 [1.11100436e-06 1.09770796e-06 1.11100436e-06 ... 3.26051871e-06
  1.18465513e-04 1.58678577e-04]
 [0.00000000e+00 0.00000000e+00 1.08683957e-01 ... 6.52103741e-06
  1.08683957e-06 6.47756383e-04]]
"""

normalized_test_data_infection_min = np.min(normalized_test_data_infection)
print("normalized_test_data_infection_min:",normalized_test_data_infection_min)
normalized_test_data_infection_max = np.max(normalized_test_data_infection)
print("normalized_test_data_infection_max:",normalized_test_data_infection_max)


raise Exception("maggie stop")

art_classifier = KerasClassifier(model=infection_classifier, clip_values=(-100, 1), use_logits=False)
pgd_attack = ProjectedGradientDescent(estimator=art_classifier, eps=0.1, eps_step=0.05, max_iter=10)





# generate adversarial example
adv_test_set = pgd_attack.generate(x=test_data_infection)

print("adv_test_set.shape:",adv_test_set.shape)

# evaluate
predictions = np.argmax(art_classifier.predict(adv_test_set), axis=1)
accuracy = np.sum(predictions == np.argmax(test_label_infection, axis=1)) / len(test_label_infection)
print(f"Accuracy of infection detector on adversarial test set: {accuracy * 100:.2f}%")

raise Exception("maggie")
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

    # ps_attack.classifier.save('/home/huan1932/IoTDataGenerate/data_preprocess/result/model/vanilla_attack_detector.h5')
    # ps_recon.save('/home/huan1932/IoTDataGenerate/data_preprocess/result/model/vanilla_reconnaissance_detector.h5')
    # ps_infec.save('/home/huan1932/IoTDataGenerate/data_preprocess/result/model/vanilla_infection_detector.h5')
    # raise Exception("maggie stop")
#------------------maggie 20230507------------------

print("Evaluate performance of vanilla infection detector")
# # 保存模型
# ps_infec.save('/home/huan1932/IoTDataGenerate/data_preprocess/result/model/vanilla_infection_detector.h5')

# 加载模型
from keras.models import load_model
ps_infec = load_model('/home/huan1932/IoTDataGenerate/data_preprocess/result/model/vanilla_infection_detector.h5')

print(" finished saving infection model")








#---------------------------------------------------
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









