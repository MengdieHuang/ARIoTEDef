#written by Zilin Shen and Daniel de Mello
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
tf.compat.v1.disable_eager_execution()
import random
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics 
from models.utils import recall_th_99, precision_th_99
import logging
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier

TIME_STEP = 2
THRESHOLD = 0.5


seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def evaluate(kerasmodel, testset, truelabel):
    pred = list(kerasmodel.predict(testset))

    fpr, tpr, thresholds_roc = metrics.roc_curve(truelabel, np.array(pred).squeeze(), pos_label=1)
    precision, recall, thresholds_pr= metrics.precision_recall_curve(truelabel, np.array(pred).squeeze(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    auprc = metrics.auc(recall, precision)

    precision99 = precision_th_99(np.array(pred).squeeze(), truelabel)
    recall99 = recall_th_99(np.array(pred).squeeze(), truelabel)

    predicts = []
    for p in pred:
        ret = (p[0] > THRESHOLD).astype("int32")
        logging.info("direct calculation number:",p[0],"predict label:",ret)
        predicts.append(ret)
    pred = np.array(predicts)
    pred = pred.reshape((pred.shape[0]),)

    if fallback:
        logging.debug("lstm> label: {}, pred: {}, ret: {}, time_step: 1".format(truelabel, pred, ret))
    else:
        logging.debug("lstm> label: {}, pred: {}, ret: {}, time_step: {}".format(truelabel, pred, ret, TIME_STEP))

    tn, fp, fn, tp = confusion_matrix(truelabel, pred).ravel()

    acc = (tn+tp)/len(truelabel)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1 = (2*precision*recall)/(precision+recall)
    #print("auc:",auc, "accuracy:", acc,"precision:", precision, "recall:", recall, "f1:",f1)
    #print("fp:",fp,",tp:",tp,",fn:",fn,",tn:",tn)

    metrics_dic = { 'auc': auc,
                    'accuracy': acc, 
                    'precision': precision, 
                    "recall": recall,
                    "f1": f1,
                    "fp": fp,
                    "tp": tp,
                    "fn": fn,
                    "tn": tn,
                    "auprc": auprc,
                    "precision99": precision99,
                    "recall99": recall99
                    }    
    return metrics_dic

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
# train_label_infection = np.load('preprocessed/train_label_infection.npy')
test_data_infection = np.load('preprocessed/test_data_infection.npy')
# train_data_infection = np.load('preprocessed/train_data_infection.npy')

# test_label_reconnaissance = np.load('preprocessed/test_label_reconnaissance.npy')
# train_label_reconnaissance = np.load('preprocessed/train_label_reconnaissance.npy')
# test_data_reconnaissance = np.load('preprocessed/test_data_reconnaissance.npy')
# train_data_reconnaissance = np.load('preprocessed/train_data_reconnaissance.npy')

# test_label_attack = np.load('preprocessed/test_label_attack.npy')
# train_label_attack = np.load('preprocessed/train_label_attack.npy')
# test_data_attack = np.load('preprocessed/test_data_attack.npy')
# train_data_attack = np.load('preprocessed/train_data_attack.npy')

# all_data = {"infection": 
#                         {
#                         'train': [train_data_infection, train_label_infection], 
#                         'test': [test_data_infection, test_label_infection]
#                         },
#             "attack": 
#                         {
#                         'train': [train_data_attack, train_label_attack], 
#                         'test': [test_data_attack, test_label_attack]
#                         },
#             "reconnaissance": 
#                         {
#                         'train': [train_data_reconnaissance, train_label_reconnaissance], 
#                         'test': [test_data_reconnaissance, test_label_reconnaissance]
#                         }
#             }

# print("train_data_infection.shape:",train_data_infection.shape)
# print("train_label_infection.shape:",train_label_infection.shape)
# print("train_data_infection:",train_data_infection)
print("test_data_infection.shape:",test_data_infection.shape)
print("test_label_infection.shape:",test_label_infection.shape)
# print("test_data_infection:",test_data_infection)

""" 
train_data_infection.shape: (19818, 41)
train_label_infection.shape: (19818,)
test_data_infection.shape: (4233, 41)
test_label_infection.shape: (4233,)
"""

test_data_infection_min = np.min(test_data_infection)
test_data_infection_max = np.max(test_data_infection)
print("test_data_infection_min:",test_data_infection_min)
print("test_data_infection_max:",test_data_infection_max)

""" 
test_data_infection_min: -1.0
test_data_infection_max: 920098.0
test_data_infection.shape: (4233, 41)
"""

print("test_data_infection.shape:",test_data_infection.shape)
# print("test_data_infection:",test_data_infection)

scale = StandardScaler().fit(test_data_infection)
#self.scale = StandardScaler().fit(dataset)
test_data_infection = scale.transform(test_data_infection)

test_data_infection_min = np.min(test_data_infection)
test_data_infection_max = np.max(test_data_infection)
print("test_data_infection_min:",test_data_infection_min)
print("test_data_infection_max:",test_data_infection_max)
print("test_data_infection.shape:",test_data_infection.shape)
# print("test_data_infection:",test_data_infection)
""" 
test_data_infection_min: -3.30513966135273
test_data_infection_max: 21.144568401380717
test_data_infection.shape: (4233, 41)
"""
fallback = False

# just use the simpler method
try:
    test_data_infection = test_data_infection.reshape((test_data_infection.shape[0], TIME_STEP, int(test_data_infection.shape[1] / TIME_STEP)))
except:
    fallback = True
    test_data_infection = test_data_infection.reshape((test_data_infection.shape[0], 1, test_data_infection.shape[1]))

print("test_data_infection.shape:",test_data_infection.shape)
# test_data_infection.shape: (4233, 1, 41)

infection_detector = load_model('/home/huan1932/IoTDataGenerate/data_preprocess/result/model/vanilla_ps-detector-infec__detector_trained_with_default_normalization.h5')
# print("infection_detector.summary():",infection_detector.summary())

print("finished loading vanilla infection detector")
print("evaluate the performance of the vanilla infection detector on clean test")

print("---------------evaluate on clean test set---------------")
cle_metrics_dic = evaluate(kerasmodel=infection_detector, testset=test_data_infection, truelabel=test_label_infection)
print("Clean Metrics: \n", cle_metrics_dic)
    
#----------------------------------    


art_classifier = KerasClassifier(model=infection_detector, clip_values=(test_data_infection_min, test_data_infection_max), use_logits=False)

# attack budget 1
# pgd_attack = ProjectedGradientDescent(estimator=art_classifier, eps=0.3, eps_step=0.1, max_iter=10)
# print("eps=0.3, eps_step=0.1, max_iter=10")

# # attack budget 2
# pgd_attack = ProjectedGradientDescent(estimator=art_classifier, eps=0.5, eps_step=0.1, max_iter=10)
# print("eps=0.5, eps_step=0.1, max_iter=10")


# attack budget 3
pgd_attack = ProjectedGradientDescent(estimator=art_classifier, eps=1, eps_step=0.5, max_iter=20)
print("eps=1.0, eps_step=0.5, max_iter=20")


# generate adversarial example
adv_test_set = pgd_attack.generate(x=test_data_infection)
print("adv_test_set.shape:",adv_test_set.shape)
# adv_test_set.shape: (4233, 1, 41)

# evaluate
print_header("Measureing vanilla Infection detector performance on Adv test data")
adv_test_set_min = np.min(adv_test_set)
adv_test_set_max = np.max(adv_test_set)
print(f"adv infection testset_min:{adv_test_set_min}")
print(f"adv infection testset_max:{adv_test_set_max}")

print("---------------evaluate on adv test set---------------") 
adv_metrics_dic = evaluate(kerasmodel=infection_detector, testset=adv_test_set, truelabel=test_label_infection)
print("Adversarial Metrics: \n", adv_metrics_dic)

raise Exception("maggie stop here")
