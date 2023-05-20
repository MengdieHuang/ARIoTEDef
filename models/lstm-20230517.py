# written by Hyunwoo, modified by Zilin
# this is for LSTM training

import sys
import copy
import logging
import numpy as np
from algorithm import Algorithm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import sklearn.metrics as metrics 
import tensorflow as tf
import sklearn as sk
from models.utils import recall_th_99, precision_th_99
from keras.callbacks import EarlyStopping

#--------maggie add packages-------
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from keras.layers import Flatten
#----------------------------------

TIME_STEP = 2
THRESHOLD = 0.5

class Lstm(Algorithm):
    def __init__(self, name):
        super().__init__(name)
        #self.data = data
        #self.label = label

    def add_dataset(self, dataset):
        self.dataset = dataset
    # Please implement the following functions
    # Concerning dataset, refer to the class TrainingSet
    
    def learning(self,features,dataset,label,kind, epochs=50, patience=None):

        self.scale = StandardScaler().fit(dataset)
        dataset = self.scale.transform(dataset)
        
        print("kind:",kind)
        trainset_min = np.min(dataset)
        trainset_max = np.max(dataset)
        print(f"{self.get_name()} {kind} trainset_min:{trainset_min}")
        print(f"{self.get_name()} {kind} trainset_max:{trainset_max}")      
        self.cle_trainset_min = trainset_min
        self.cle_trainset_max = trainset_max
        
        fallback = False

        # just use the simpler method
        try:
            dataset = dataset.reshape((dataset.shape[0], TIME_STEP, int(dataset.shape[1] / TIME_STEP)))
        except:
            fallback = True
            dataset = dataset.reshape((dataset.shape[0], 1, dataset.shape[1]))

        labels = label

        self.classifier[kind] = Sequential()
        if fallback:
            self.classifier[kind].add(LSTM(128, return_sequences=True, activation='relu', input_shape=(1, features)))
        else:
            self.classifier[kind].add(LSTM(128, return_sequences=True, activation='relu', input_shape=(TIME_STEP, int(features / TIME_STEP))))
        self.classifier[kind].add(LSTM(128, return_sequences=True, activation='relu'))
        self.classifier[kind].add(Dense(1, activation='sigmoid'))
        self.classifier[kind].add(Flatten())
        
        
        metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        self.classifier[kind].compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

        print("the info of dataset," ,dataset.shape)

        es = EarlyStopping(monitor='val_loss', verbose=1, patience=patience)
        try:
            print("shuffle is enabled for training")
            dataset, labels = shuffle(dataset,labels)
            if patience is None:
                print("Early stopping: NO.")
                self.classifier[kind].fit(dataset, labels, epochs=epochs, validation_split=0.1, verbose=2)
            else:
                print("Early stopping: YES.")
                self.classifier[kind].fit(dataset, labels, epochs=epochs, validation_split=0.1, verbose=2, callbacks=[es])
            if fallback:
                logging.info("{} {} classifier is generated with the time step 1".format(self.get_name(), kind))
                print("classifier is generated with time step 1")
            else:
                logging.info("{} {} classifier is generated with the time step {}".format(self.get_name(), kind, TIME_STEP))
                print("classifire is generates with time step",TIME_STEP)
        except:
            self.classifier[kind] = None
            logging.info("{} {} classifier is not generated".format(self.get_name(), kind))


    def cal_fitness(self, dataset, label, kind):
        test = np.array(dataset)
        test = self.scale.transform(test)
        fallback = False
        try:
            test = test.reshape((test.shape[0], TIME_STEP, int(test.shape[1] / TIME_STEP)))
        except:
            fallback = True
            test = test.reshape((test.shape[0],1,test.shape[1]))

        pred = list(self.classifier[kind].predict(test))
        return pred


    def predict(self, dataset, kind):
        #logging.debug("window.get_code(): {}".format(window.get_code()))
        #label = window.get_label(kind)
        #test = window.get_code().copy()
        test = np.array(dataset)
        test = self.scale.transform(test)
        fallback = False
        try:
            test = test.reshape((test.shape[0], TIME_STEP, int(test.shape[1] / TIME_STEP)))
        except:
            fallback = True
            test = test.reshape((test.shape[0], 1, test.shape[1]))

        pred = list(self.classifier[kind].predict(test))
        return pred

    def detection(self, dataset, label, kind):
        #logging.debug("window.get_code(): {}".format(window.get_code()))
        #label = window.get_label(kind)
        #test = window.get_code().copy()
        test = np.array(dataset)
        test = self.scale.transform(test)

        #------------maggie add---------------
        print("kind:",kind)        
        testset_min = np.min(test)
        testset_max = np.max(test)
        print(f"{self.get_name()} {kind} testset_min:{testset_min}")
        print(f"{self.get_name()} {kind} testset_max:{testset_max}")      
        self.cle_testset_min = testset_min
        self.cle_testset_max = testset_max
        #-------------------------------------
                
        
        fallback = False
        try:
            test = test.reshape((test.shape[0], TIME_STEP, int(test.shape[1] / TIME_STEP)))
        except:
            fallback = True
            test = test.reshape((test.shape[0], 1, test.shape[1]))

        pred = list(self.classifier[kind].predict(test))
        fpr, tpr, thresholds_roc = metrics.roc_curve(label, np.array(pred).squeeze(), pos_label=1)
        precision, recall, thresholds_pr= metrics.precision_recall_curve(label, np.array(pred).squeeze(), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        auprc = metrics.auc(recall, precision)
        
        precision99 = precision_th_99(np.array(pred).squeeze(), label)
        recall99 = recall_th_99(np.array(pred).squeeze(), label)

        predicts = []
        for p in pred:
            ret = (p[0] > THRESHOLD).astype("int32")
            logging.info("direct calculation number:",p[0],"predict label:",ret)
            predicts.append(ret)
        pred = np.array(predicts)
        pred = pred.reshape((pred.shape[0]),)

        if fallback:
            logging.debug("lstm> label: {}, pred: {}, ret: {}, time_step: 1".format(label, pred, ret))
        else:
            logging.debug("lstm> label: {}, pred: {}, ret: {}, time_step: {}".format(label, pred, ret, TIME_STEP))

        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(label, pred).ravel()

        acc = (tn+tp)/len(label)
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
        return pred, acc, metrics_dic


    #------------------maggie add------------------
    def advgenerate(self, cle_testset_x, cle_testset_y, kind, args):        
        cle_testset_x = np.array(cle_testset_x)
        cle_testset_x = self.scale.transform(cle_testset_x)        

        print("kind:",kind)        
        cle_testset_min = np.min(cle_testset_x)
        cle_testset_max = np.max(cle_testset_x)
        print(f"{self.get_name()} {kind} cle_testset_min:{cle_testset_min}")
        print(f"{self.get_name()} {kind} cle_testset_max:{cle_testset_max}")      
        self.cle_testset_min = cle_testset_min
        self.cle_testset_max = cle_testset_max

        print("cle_testset_x.shape:",cle_testset_x.shape)
        
        """ 
        kind:
        ps-detector-recon  cle_testset_min:-2.4484992496151756
        ps-detector-recon  cle_testset_max:28.32679285519669
        cle_testset_x.shape: (4233, 41)
        """
        
        fallback = False
        try:
            cle_testset_x = cle_testset_x.reshape((cle_testset_x.shape[0], TIME_STEP, int(cle_testset_x.shape[1] / TIME_STEP)))
        except:
            fallback = True
            cle_testset_x = cle_testset_x.reshape((cle_testset_x.shape[0], 1, cle_testset_x.shape[1]))        
        print("cle_testset_x.shape:",cle_testset_x.shape)
        """ 
        cle_testset_x.shape: (4233, 1, 41)
        """

        art_classifier = KerasClassifier(model=self.classifier[kind], clip_values=(self.cle_testset_min, self.cle_testset_max), use_logits=False)

        pgd_attack = ProjectedGradientDescent(estimator=art_classifier, eps=args.eps, eps_step=args.eps_step, max_iter=args.max_iter)
        print(f'eps={args.eps},eps_step={args.eps_step},max_iter={args.max_iter}')

        adv_testset_x = pgd_attack.generate(x=cle_testset_x)
        adv_testset_y = cle_testset_y
        
        print("adv_testset_x.shape:",adv_testset_x.shape)
        print("adv_testset_y.shape:",adv_testset_y.shape)
        """ 
        adv_test_examples.shape: (4233, 1, 41)
        adv_test_labels.shape: (4233,)
        """
        
        adv_testset_x = adv_testset_x.reshape((adv_testset_x.shape[0],cle_testset_x.shape[2]))
        print("adv_testset_x.shape:",adv_testset_x.shape)
        # adv_testset_x.shape: (4233, 41)
        
        return adv_testset_x, adv_testset_y     
    
    def advgenerate_onlymail(self, cle_testset_x, cle_testset_y, kind, args):        
        cle_testset_x = np.array(cle_testset_x)
        cle_testset_x = self.scale.transform(cle_testset_x)        

        print("kind:",kind)        
        cle_testset_min = np.min(cle_testset_x)
        cle_testset_max = np.max(cle_testset_x)
        print(f"{self.get_name()} {kind} cle_testset_min:{cle_testset_min}")
        print(f"{self.get_name()} {kind} cle_testset_max:{cle_testset_max}")      
        self.cle_testset_min = cle_testset_min
        self.cle_testset_max = cle_testset_max

        print("cle_testset_x.shape:",cle_testset_x.shape)
        print("cle_testset_y.shape:",cle_testset_y.shape)
        print("cle_testset_y[:10]:",cle_testset_y[:10])
        
        """
        kind:
        ps-detector-infec  cle_testset_min:-1.7222021363244926
        ps-detector-infec  cle_testset_max:31.46267630066733
        cle_testset_x.shape: (4233, 41)
        cle_testset_y.shape: (4233,)
        cle_testset_y[:10]: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        """
                
        condition = cle_testset_y.astype(bool)
        print("condition.shape:",condition.shape)
        print("condition[:10]:",condition[:10])
        # adv_malicious_y = torch.masked_select(cle_testset_y, condition)
        malicious_cle_testset_y = np.extract(condition,cle_testset_y)
        print("malicious_cle_testset_y.shape:",malicious_cle_testset_y.shape)
                                
        """ 
        condition.shape: (4233,)
        condition[:10]: [False False False False False False False False False False]
        malicious_cle_testset_y.shape: (318,)
        """
        
        # malicious_cle_testset_x = np.extract(condition,cle_testset_x)
        cond=np.expand_dims(condition,1)
        print("cond.shape:",cond.shape)
        # 创建形状为(4233, 41)的全False数组
        cond_expend = np.full((cle_testset_x.shape[0], cle_testset_x.shape[1]), False, dtype=bool)
        # 将条件数组广播到result数组中
        cond = np.logical_or(cond_expend, cond)        
        print("cond.shape:",cond.shape)        
        """
        condition.shape: (4233,)
        condition[:10]: [False False False False False False False False False False]
        malicious_cle_testset_y.shape: (318,)
        cond.shape: (4233, 1)
        cond.shape: (4233, 41)
        """
        
        malicious_cle_testset_x = np.extract(cond,cle_testset_x)
        print("malicious_cle_testset_x.shape:",malicious_cle_testset_x.shape)        
        malicious_cle_testset_x = np.reshape(malicious_cle_testset_x, (malicious_cle_testset_y.shape[0], cle_testset_x.shape[1]))        
        print("malicious_cle_testset_x.shape:",malicious_cle_testset_x.shape)
        """ 
        malicious_cle_testset_x.shape: (13038,)
        malicious_cle_testset_x.shape: (318, 41)
        """
        # raise Exception("maggie stop here!!!!")
        
        fallback = False
        try:
            malicious_cle_testset_x = malicious_cle_testset_x.reshape((malicious_cle_testset_x.shape[0], TIME_STEP, int(malicious_cle_testset_x.shape[1] / TIME_STEP)))
        except:
            fallback = True
            malicious_cle_testset_x = malicious_cle_testset_x.reshape((malicious_cle_testset_x.shape[0], 1, malicious_cle_testset_x.shape[1]))        
        print("malicious_cle_testset_x.shape:",malicious_cle_testset_x.shape)
        """ 
        """

        art_classifier = KerasClassifier(model=self.classifier[kind], clip_values=(self.cle_testset_min, self.cle_testset_max), use_logits=False)

        pgd_attack = ProjectedGradientDescent(estimator=art_classifier, eps=args.eps, eps_step=args.eps_step, max_iter=args.max_iter)
        print(f'eps={args.eps},eps_step={args.eps_step},max_iter={args.max_iter}')

        adv_testset_x = pgd_attack.generate(x=malicious_cle_testset_x)
        adv_testset_y = malicious_cle_testset_y
        
        print("adv_testset_x.shape:",adv_testset_x.shape)
        print("adv_testset_y.shape:",adv_testset_y.shape)
        """ 
        adv_testset_x.shape: (318, 1, 41)
        adv_testset_y.shape: (318,)
        """
        
        adv_testset_x = adv_testset_x.reshape((adv_testset_x.shape[0],adv_testset_x.shape[2]))
        print("adv_testset_x.shape:",adv_testset_x.shape)
        # adv_testset_x.shape: (318, 41)
        
        return adv_testset_x, adv_testset_y     
        
    #--------------------------------------------------