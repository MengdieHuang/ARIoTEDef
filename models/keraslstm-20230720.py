from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_NUMA_NODES'] = '1'
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
import numpy as np
import time
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

from keras.layers import RepeatVector, TimeDistributed, BatchNormalization, Activation, Input, dot, concatenate, Attention
from tensorflow.keras.optimizers import Adam
from keras.models import Model    
from tensorflow.keras import optimizers

import sys
sys.stdout.flush()
print("正在导入art库...", flush=True)
from art.estimators.classification.keras import KerasClassifier
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
print("art库导入完成!", flush=True) 

def calculate_tp_tn_fp_fn(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1

    return tp, tn, fp, fn

        
def seq2seq_model(input_shape, output_shape, hidden_units):
    train_input = Input(shape=input_shape)
    train_output = Input(shape=output_shape)    
    print("train_input.shape:", train_input.shape)
    print("train_output.shape:", train_output.shape)
    
    encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
            units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
            return_sequences=True, return_state=True)(train_input)

    encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
    encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

    decoder_input = RepeatVector(train_output.shape[1])(encoder_last_h)
    decoder_stack_h = LSTM(units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
            return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
    
    
    print("encoder_stack_h.shape:",encoder_stack_h.shape)
    print("decoder_stack_h.shape:",decoder_stack_h.shape)
    
    attention = dot([decoder_stack_h, encoder_stack_h], axes=[2,2])
    attention = Activation('softmax')(attention)

    context = dot([attention, encoder_stack_h], axes=[2,1])
    context = BatchNormalization(momentum=0.6)(context)

    decoder_combined_context = concatenate([context, decoder_stack_h])
    out = TimeDistributed(Dense(train_output.shape[2]))(decoder_combined_context)
    out = Activation('sigmoid')(out)

    print("train_input.shape:",train_input.shape)                
    print("out.shape:",out.shape)       
    
    """ 
    encoder_stack_h.shape: (None, 10, 128)
    decoder_stack_h.shape: (None, 10, 128)
    input_train.shape: (None, 10, 4)
    out.shape: (None, 10, 1)
    """ 
    
    model = Model(inputs=train_input, outputs=out)
    # self.model.summary()   
    
    return model

        
class EpochTimer(Callback):
    def __init__(self):
        self.epoch_times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        self.epoch_times.append(elapsed_time)

class PSDetector():
    def __init__(self, name, args):
        self.modelname = name
        self.args = args
        
    def add_dataset(self, dataset):
        self.dataset = dataset

        self.trainset_min = np.min(self.dataset['train'][0])
        self.trainset_max = np.max(self.dataset['train'][0])
        self.testset_min = np.min(self.dataset['test'][0])
        self.testset_max = np.max(self.dataset['test'][0])    
            
        print(f"{self.modelname} trainset_min:{self.trainset_min:.4f}")
        print(f"{self.modelname} trainset_max:{self.trainset_max:.4f}")
        print(f"{self.modelname} testset_min:{self.testset_min:.4f}")
        print(f"{self.modelname} testset_max:{self.testset_max:.4f}")            
                
    def def_model(self, input_dim=41, output_dim=1, timesteps=1):  

        model = Sequential()

        model.add(LSTM(units=128, activation='relu', return_sequences=True, input_shape=(timesteps, int(input_dim / timesteps))))
        model.add(LSTM(units=128, activation='relu', return_sequences=True))                        
        # 输出128维
        model.add(Dense(units=output_dim, activation='sigmoid'))
        # 输出1维
        model.add(Flatten())
        self.model = model

    def stdtrain(self, timesteps, exp_result_dir):
        
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")
   
        trainset_x = self.dataset['train'][0]
        trainset_y = self.dataset['train'][1]
  
        trainset_x, trainset_y = shuffle(trainset_x, trainset_y)
        
        trainset_x = trainset_x.reshape((trainset_x.shape[0], timesteps, int(math.ceil(trainset_x.shape[1] / timesteps))))

        # print("trainset_x.shape:",trainset_x.shape)
        """
        trainset_x.shape: (19152, 1, 41)
        """
        
        # 配置模型的训练过程
        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=self.args.lr), metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=self.args.patience, verbose=1)    
        timer_callback = EpochTimer()
        callbacks = [early_stop, timer_callback]
        history=self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.batchsize, epochs=self.args.ps_epochs, verbose=2, callbacks=callbacks, validation_split=0.2)       


        epo_train_loss = history.history['loss']
        epo_val_loss = history.history['val_loss']
        epo_train_acc = history.history['accuracy']
        epo_val_acc = history.history['val_accuracy']
        epo_cost_time = timer_callback.epoch_times

        # 将准确率历史记录转换为百分比
        epo_train_acc = [accuracy * 100 for accuracy in epo_train_acc]
        epo_val_acc = [accuracy * 100 for accuracy in epo_val_acc]

        #--------save plt---------            
        loss_png_name = f'Loss of standard trained {self.modelname}'
        accuracy_png_name = f'Accuracy of standard trained {self.modelname}'        
        time_png_name = f'Cost time of standard trained {self.modelname}'
         
        plt.style.use('seaborn')           
        plt.plot(list(range(len(epo_train_loss))), epo_train_loss, label='Train Loss', marker='o')
        # plt.plot(epo_val_loss, label='Validation Loss', marker='s')
        plt.plot(list(range(len(epo_val_loss))), epo_val_loss, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{loss_png_name}')
        # plt.show()
        plt.savefig(f'{exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(list(range(len(epo_train_acc))), epo_train_acc, label='Train Accuracy', marker='o')
        plt.plot(list(range(len(epo_val_acc))), epo_val_acc, label='Validation Accuracy', marker='s')        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        # plt.ylim(0, 1)
        plt.legend(loc='best',frameon=True)
        plt.title(f'{accuracy_png_name}')        
        # plt.show()
        plt.savefig(f'{exp_result_dir}/{accuracy_png_name}.png')
        plt.close()

        plt.plot(list(range(len(epo_cost_time))), epo_cost_time, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cost Time')
        # plt.legend(loc='best',frameon=True)
        plt.title(f'{time_png_name}')        
        # plt.show()
        plt.savefig(f'{exp_result_dir}/{time_png_name}.png')
        plt.close()
             
    def evaluate(self, testset_x, testset_y):
        
        test_los, _ = self.model.evaluate(testset_x, testset_y)
        output = self.model.predict(testset_x)
        # print("output:",output)
        # print("output.shape:",output.shape)
        
        predicts = []
        for p in output:
            ret = (p[0] > 0.5).astype("int32")
            # print(f"direct calculation number:{p[0]}, predict label:{ret}")
            predicts.append(ret)
            
        output = np.array(predicts)
        
        print("confusion_matrix(testset_y, output).ravel():",confusion_matrix(testset_y, output).ravel())
        
        # test_TN, test_FP, test_FN, test_TP = confusion_matrix(testset_y, output).ravel()
        # test_TN, test_FP, test_FN, test_TP = calculate_tp_tn_fp_fn(y_true=testset_y, y_pred=output)
        test_TP, test_TN, test_FP, test_FN = calculate_tp_tn_fp_fn(y_true=testset_y, y_pred=output)
        
        test_acc = accuracy_score(testset_y, output)
        test_recall = recall_score(testset_y, output, average='macro')
        test_precision = precision_score(testset_y, output, average='macro')
        test_F1 = f1_score(testset_y, output, average='macro')
        
        # test_FPR = test_FP / (test_FP + test_TN)
        
        # return test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1
    
        # return test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1
    
        return round(test_acc, 4), round(test_los, 4), test_TP, test_FP, test_TN, test_FN, round(test_recall, 4), round(test_precision, 4), round(test_F1, 4)
    
    def test(self, testset_x, testset_y, timesteps):
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        # print(f"prepare test set for evaluating {self.modelname} ")
        # testset_x = self.dataset['test'][0]
        # testset_y = self.dataset['test'][1]    
        # print("testset_x.shape:",testset_x.shape)
        # print("testset_y.shape:",testset_y.shape)
        
        testset_x = testset_x.reshape((testset_x.shape[0], timesteps, int(math.ceil(testset_x.shape[1] / timesteps))))
        # print("testset_x.shape:",testset_x.shape)
        
        """ 
        testset_x.shape: (4233, 41)
        testset_y.shape: (4233,)
        testset_x.shape: (4233, 1, 41)
        """
    
        # test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1 = self.evaluate(testset_x, testset_y)
        
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = self.evaluate(testset_x, testset_y)

        
        return test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1
             
    def generate_advmail(self,timesteps):        
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        # print(f"prepare test set for generating adversarial testset against {self.modelname} ")
        cle_testset_x = self.dataset['test'][0]
        cle_testset_y = self.dataset['test'][1]    
        # print("cle_testset_x.shape:",cle_testset_x.shape)
        # print("cle_testset_y.shape:",cle_testset_y.shape)
        # print("cle_testset_y[:10]:",cle_testset_y[:10])        
        """
        cle_testset_x.shape: (4224, 41)
        cle_testset_y.shape: (4224,)
        cle_testset_y[:10]: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        """
        # print(f"{self.modelname} cle_testset_min:{self.testset_min}")
        # print(f"{self.modelname} cle_testset_max:{self.testset_max}")      
       
        """ 
        attack-detector cle_testset_min:-3.30513966135273
        attack-detector cle_testset_max:21.144568401380717
        """
        # benign 0 malicious 1
        
        
        
        # extract malicious set
        condition = cle_testset_y.astype(bool)
        # print("condition.shape:",condition.shape)
        # print("condition[:10]:",condition[:10])
        
        malicious_cle_testset_y = np.extract(condition,cle_testset_y)
        print("malicious_cle_testset_y.shape:",malicious_cle_testset_y.shape)
        print("malicious_cle_testset_y:",malicious_cle_testset_y)
                                
        """ 
        condition.shape: (4233,)
        condition[:10]: [False False False False False False False False False False]
        malicious_cle_testset_y.shape: (123,)
        malicious_cle_testset_y: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        """
        
        cond=np.expand_dims(condition,1)
        # print("cond.shape:",cond.shape)
        # 创建形状为(4233, 41)的全False数组
        cond_expend = np.full((cle_testset_x.shape[0], cle_testset_x.shape[1]), False, dtype=bool)
        # 将条件数组广播到result数组中
        cond = np.logical_or(cond_expend, cond)        
        # print("cond.shape:",cond.shape)        
        """
        cond.shape: (4233, 1)
        cond.shape: (4233, 41)
        """
        
        malicious_cle_testset_x = np.extract(cond,cle_testset_x)
        # print("malicious_cle_testset_x.shape:",malicious_cle_testset_x.shape)        

        """
        malicious_cle_testset_x.shape: (5043,)

        """        
        malicious_cle_testset_x = np.reshape(malicious_cle_testset_x, (malicious_cle_testset_y.shape[0], cle_testset_x.shape[1]))        
        # print("malicious_cle_testset_x.shape:",malicious_cle_testset_x.shape)
        """ 
        malicious_cle_testset_x.shape: (123, 41)
        """
        
        
        malicious_cle_testset_x = malicious_cle_testset_x.reshape((malicious_cle_testset_x.shape[0], timesteps, int(math.ceil(malicious_cle_testset_x.shape[1] / timesteps))))
        print("malicious_cle_testset_x.shape:",malicious_cle_testset_x.shape)
        
        """ 
        malicious_cle_testset_x.shape: (123, 1, 41)
        """
               

        # print("self.testset_min:",self.testset_min)
        # print("self.testset_max:",self.testset_max)
        
        # print("self.args.eps:",self.args.eps)
        # print("self.args.eps_step:",self.args.eps_step)
        # print("self.args.max_iter:",self.args.max_iter)
        
        # import sys
        # sys.stdout.flush()
        # print("正在导入art库...", flush=True)
        # from art.estimators.classification.keras import KerasClassifier
        # from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
        # print("art库导入完成!", flush=True)   
             
        art_classifier = KerasClassifier(model=self.model, clip_values=(self.testset_min, self.testset_max), use_logits=False)

        pgd_attack = ProjectedGradientDescent(estimator=art_classifier, eps=self.args.eps, eps_step=self.args.eps_step, max_iter=self.args.max_iter)

        print(f'eps={self.args.eps},eps_step={self.args.eps_step},max_iter={self.args.max_iter}')

        """ 
        self.args.eps: 1.0
        self.args.eps_step: 0.5
        self.args.max_iter: 20
        eps=1.0,eps_step=0.5,max_iter=20
        """
        adv_testset_x = pgd_attack.generate(x=malicious_cle_testset_x)
        adv_testset_y = malicious_cle_testset_y
        
        # print("adv_testset_x.shape:",adv_testset_x.shape)
        # print("adv_testset_y.shape:",adv_testset_y.shape)
        """ 
        adv_testset_x.shape: (318, 1, 41)
        adv_testset_y.shape: (318,)
        """
        
        adv_testset_x = adv_testset_x.reshape((adv_testset_x.shape[0],adv_testset_x.shape[2]))
        # print("adv_testset_x.shape:",adv_testset_x.shape)
        # adv_testset_x.shape: (318, 41)
                
    
        return adv_testset_x, adv_testset_y

    def retrain(self, retrainset_x, retrainset_y, timesteps, curround_exp_result_dir):
        
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        print(f"prepare retraining set for learning {self.modelname} ")
        
        trainset_x = retrainset_x
        trainset_y = retrainset_y

        trainset_x, trainset_y = shuffle(trainset_x, trainset_y)
        # print("trainset_x.shape:",trainset_x.shape)
        # print("trainset_y.shape:",trainset_y.shape)

        trainset_x = trainset_x.reshape((trainset_x.shape[0], timesteps, int(math.ceil(trainset_x.shape[1] / timesteps))))

        # print("trainset_x.shape:",trainset_x.shape)


        # trainset_x, valset_x, trainset_y, valset_y = train_test_split(trainset_x, trainset_y, test_size=0.1, random_state=42)
        # # print("trainset_x.shape:",trainset_x.shape)
        # # print("trainset_y.shape:",trainset_y.shape)        
        # # print("valset_x.shape:",valset_x.shape)
        # # print("valset_y.shape:",valset_y.shape)        
      
        early_stop = EarlyStopping(monitor='val_loss', patience=self.args.patience, verbose=1)    
      
        timer_callback = EpochTimer()
        callbacks = [early_stop, timer_callback]
        # history=self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.batchsize, epochs=self.args.ps_epochs, verbose=2, callbacks=callbacks, validation_data=(valset_x,valset_y))       
        history=self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.batchsize, epochs=self.args.ps_epochs, verbose=2, callbacks=callbacks, validation_split=0.2)       

        epo_train_loss = history.history['loss']
        epo_val_loss = history.history['val_loss']
        epo_train_acc = history.history['accuracy']
        epo_val_acc = history.history['val_accuracy']
        epo_cost_time = timer_callback.epoch_times

        # 将准确率历史记录转换为百分比
        epo_train_acc = [accuracy * 100 for accuracy in epo_train_acc]
        epo_val_acc = [accuracy * 100 for accuracy in epo_val_acc]
        
        
        rou_cost_time = sum(epo_cost_time)
        #--------save plt---------            
        loss_png_name = f'Loss of retrained {self.modelname}'
        accuracy_png_name = f'Accuracy of retrained {self.modelname}'        
        time_png_name = f'Cost time of retrained {self.modelname}'
             
        plt.style.use('seaborn')
           
        plt.plot(epo_train_loss, label='Train Loss', marker='o')
        plt.plot(epo_val_loss, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{loss_png_name}')
        # plt.show()
        plt.savefig(f'{curround_exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(epo_train_acc, label='Train Accuracy', marker='o')
        plt.plot(epo_val_acc, label='Validation Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        # plt.ylim(0, 1)
        plt.legend(loc='best',frameon=True)
        plt.title(f'{accuracy_png_name}')        
        # plt.show()
        plt.savefig(f'{curround_exp_result_dir}/{accuracy_png_name}.png')
        plt.close()

        plt.plot(epo_cost_time, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cost Time')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{time_png_name}')        
        # plt.show()
        plt.savefig(f'{curround_exp_result_dir}/{time_png_name}.png')
        plt.close()
        
        return rou_cost_time
        
    def load_model(self, model_path):
        from keras.models import load_model
        self.model = load_model(model_path)
        
    def save_model(self, save_path):
        self.model.save(save_path)
        
    def analysis(self, test_windows_x):
        # detector predict test_windows_x
        detector_probs = self.model.predict(test_windows_x)
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
         
        return detector_tagged_windows_probs, detector_tagged_windows_idxs

class Seq2Seq():
    def __init__(self, name, args):
        self.modelname = name
        self.args = args

    def add_dataset(self, dataset):
        self.dataset = dataset

        self.trainset_min = np.min(self.dataset['train'][0])
        self.trainset_max = np.max(self.dataset['train'][0])
        self.testset_min = np.min(self.dataset['test'][0])
        self.testset_max = np.max(self.dataset['test'][0])    
            
        print(f"{self.modelname} trainset_min:{self.trainset_min:.4f}")
        print(f"{self.modelname} trainset_max:{self.trainset_max:.4f}")
        print(f"{self.modelname} testset_min:{self.testset_min:.4f}")
        print(f"{self.modelname} testset_max:{self.testset_max:.4f}")  
    
    def probability_based_embedding(self, p, d):
        ret = 0
        pr = {}

        tmp = zip(range(4), p)
        order = [k for k, _ in sorted(tmp, key=lambda x: x[1], reverse=True)]
                    
        ru = 0
        rd = 0
        for i in range(4):
            r = round(p[i], d)
            c = math.ceil(p[i] * 10 ** d) / 10 ** d
            f = math.floor(p[i] * 10 ** d) / 10 ** d
            if r - c == 0:
                ru += 1
            elif r - f == 0:
                rd += 1

        lst = []
        if ru >= 2:
            for i in range(4):
                if i == order[-1]:
                    lst.append(math.floor(p[i] * 10 ** d) / 10 ** d)
                else:
                    lst.append(round(p[i], d))
            if sum(lst) > 0.999 and sum(lst) < 1.001:
                for i in range(4):
                    pr[i] = lst[i]
            else:
                for i in range(4):
                    if i == order[-1] or i == order[-2]:
                        pr[i] = math.floor(p[i] * 10 ** d) / 10 ** d
                    else:
                        pr[i] = round(p[i], d)

        elif rd >= 2:
            for i in range(4):
                if i == order[0]:
                    lst.append(math.ceil(p[i] * 10 ** d) / 10 ** d)
                else:
                    lst.append(round(p[i], d))
            if sum(lst) > 0.999 and sum(lst) < 1.001:
                for i in range(4):
                    pr[i] = lst[i]
            else:
                for i in range(4):
                    if i == order[0] or i == order[1]:
                        pr[i] = math.ceil(p[i] * 10 ** d) / 10 ** d
                    else:
                        pr[i] = round(p[i], d)

        for i in [2, 1, 0, 3]:
            ret *= 10 ** d
            ret += round(pr[i] * (10 ** d), 0)
        
        return ret

    def truncate(self, x, y, idxs_order, slen=100):
        in_, out_, truncated_idxs = [], [], []

        for i in range(len(x) - slen + 1):
            in_.append(x[i:i+slen])
            out_.append(y[i:(i+slen)])
            truncated_idxs.append(idxs_order[i:(i+slen)])
        return np.array(in_), np.array(out_), np.array(truncated_idxs)

    def permute_truncated(self, X_in, X_out, truncated_idxs, slen=10, inplace=False):
        enable_permute_prints = False
        if not inplace:
            X_in = copy.copy(X_in)
            truncated_idxs = copy.copy(truncated_idxs)
        for x_seq_in, x_seq_out, seq_idxs in zip(X_in, X_out, truncated_idxs):
            repeating_seq = []
            permute_idxs = []
            i = 0
            current_label = x_seq_out[i]
            #label_next = current_label
            repeating_seq.append(i)
            i+=1
            while i < slen:
                prev_label = current_label
                current_label = x_seq_out[i]
                if i < 20 and enable_permute_prints:
                    #assert(0)
                    print(i, current_label, prev_label)

                if prev_label != current_label: 
                    if i < 20 and enable_permute_prints:
                        print(repeating_seq)
                    
                    np.random.seed(self.args.seed)    
                    permute_idxs = permute_idxs + list(np.random.permutation(repeating_seq))
                    #x_seq_in[repeating_seq] = x_seq_in[idx_permutation]
                    repeating_seq = []
                    repeating_seq.append(i)
                    i+=1
                else:
                    repeating_seq.append(i)
                    i+=1 
                if i < 20 and enable_permute_prints:
                    print(repeating_seq)
            
            np.random.seed(self.args.seed)    
            permute_idxs = permute_idxs + list(np.random.permutation(repeating_seq))
            if i < 20 and enable_permute_prints:
                print("permuting {} with idxs {}".format(x_seq_in, permute_idxs))
                print("permuting {} with idxs {}".format(seq_idxs, permute_idxs))
            self.permute_sublist_inplace(x_seq_in, permute_idxs)    
            self.permute_sublist_inplace(seq_idxs, permute_idxs)
            #print(seq_idxs)
        if not inplace:
            return X_in, truncated_idxs
    
    def evaluate(self, testset_x, testset_y):
        
        

        
        # print("testset_x.shape:",testset_x.shape)
        # print("testset_y.shape:",testset_y.shape)
        # print("testset_y[:1]:",testset_y[:1])
        """ 
        testset_x.shape: (4215, 10, 4)
        testset_y.shape: (4215, 10, 1)
        """
        
                
        output = self.model.predict(testset_x)
        # print("output.shape:",output.shape)
        print("output[:1]:",output[:1])
        
        """ 
        output.shape: (4215, 10, 1)
        """
         
       
        y_pred_binary = np.round(output).astype(int)
        y_test_binary = np.round(testset_y).astype(int)       
        
        # print("y_pred_binary.shape:",y_pred_binary.shape)
        # print("y_pred_binary:",y_pred_binary)
        # print("y_test_binary.shape:",y_test_binary.shape)                  
        # print("y_test_binary:",y_test_binary)                  
        
        """ 
        y_pred_binary.shape: (4215, 10, 1)
        y_pred_binary: [[[0.]

        y_test_binary.shape: (4215, 10, 1)
        y_test_binary: [[[0.]
        """   
        

        y_test_binary_2d = y_test_binary.squeeze()
        y_pred_binary_2d = y_pred_binary.squeeze()
        
        # print("y_test_binary_2d.shape:",y_test_binary_2d.shape)  
        # print("y_pred_binary_2d.shape:",y_pred_binary_2d.shape)  
        
        """ 
        y_test_binary_2d.shape: (4215, 10)
        """
        
        y_test_binary_1d = []
        for row in y_test_binary_2d:
            counts = np.bincount(row)
            most_frequent_index = np.argmax(counts)
            y_test_binary_1d.append(most_frequent_index)
        y_test_binary_1d = np.array(y_test_binary_1d)

        # print("y_test_binary_1d:",y_test_binary_1d)  # (4215,)
        # print("y_test_binary_1d.shape:",y_test_binary_1d.shape)  # (4215,)
        

        y_pred_binary_1d = []
        for row in y_pred_binary_2d:
            counts = np.bincount(row)
            most_frequent_index = np.argmax(counts)
            y_pred_binary_1d.append(most_frequent_index)
        y_pred_binary_1d = np.array(y_pred_binary_1d)
                
        # print("y_pred_binary_1d:",y_pred_binary_1d)  # (4215,)                
        # print("y_pred_binary_1d.shape:",y_pred_binary_1d.shape)  # (4215,)        
        
        """ 
        y_test_binary_2d.shape: (4215, 10)
        y_pred_binary_2d.shape: (4215, 10)
        y_test_binary_1d: [0 0 0 ... 0 0 0]
        y_test_binary_1d.shape: (4215,)
        y_pred_binary_1d: [0 0 0 ... 0 0 0]
        y_pred_binary_1d.shape: (4215,)
        """

        print("confusion_matrix(y_test_binary_1d, y_pred_binary_1d).ravel():",confusion_matrix(y_test_binary_1d, y_pred_binary_1d).ravel())
        # test_TN, test_FP, test_FN, test_TP = confusion_matrix(testset_y, output).ravel()
        # test_TN, test_FP, test_FN, test_TP = confusion_matrix(y_test_binary_1d, y_pred_binary_1d).ravel()
        test_TP, test_TN, test_FP, test_FN = calculate_tp_tn_fp_fn(y_true=y_test_binary_1d, y_pred=y_pred_binary_1d)


        test_acc = accuracy_score(y_test_binary_1d, y_pred_binary_1d)
        print(f"Test accuracy: {100*test_acc} %")
        test_recall = recall_score(y_test_binary_1d, y_pred_binary_1d, average='macro')
        test_precision = precision_score(y_test_binary_1d, y_pred_binary_1d, average='macro')
        test_F1 = f1_score(y_test_binary_1d, y_pred_binary_1d, average='macro')
        # test_FPR = test_FP / (test_FP + test_TN)

        
        # test_los, test_acc_v2 = self.model.evaluate(x=testset_x, y=testset_y)
        test_los, _ = self.model.evaluate(x=testset_x, y=y_test_binary)

        # print("test_acc_v2:",test_acc_v2)

        
        
        # return test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1
    
        return round(test_acc, 4), round(test_los, 4), test_TP, test_FP, test_TN, test_FN, round(test_recall, 4), round(test_precision, 4), round(test_F1, 4)
      
    def test(self, events, labels):
        print("events.shape:",events.shape)
        print("labels.shape:",labels.shape)  
              
        slen = self.args.sequence_length

        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        # in_, out_ = [], []
        # idx_order = []
        # idx = 0

        testset_x, testset_y = [], []
        idx_order = []
        idx = 0        

        
        for idx, (event, label) in enumerate(zip(events, labels)):
            
            testset_x.append(event)
            testset_y.append([label])
            idx_order.append(idx)
                    
        testset_x, testset_y, truncated_idxs = self.truncate(testset_x, testset_y, idx_order, slen=slen)
        # X_out_labels = np.array(X_out)[:,:,0].tolist()


        print("testset_x.shape:", testset_x.shape)
        print("testset_y.shape:", testset_y.shape)

        # test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = self.evaluate(X_in, X_out[:, :, :1])
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = self.evaluate(testset_x, testset_y)
        
        
        return test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1
            
    def analysis(self, events, labels):
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        slen = self.args.sequence_length
        # print("self.args.sequence_length:",self.args.sequence_length)
        # self.args.sequence_length: 10


        # print(f"prepare test set for analysis {self.modelname} ")
  
        # print("events.shape:", events.shape)
        # print("labels.shape:", labels.shape)
        """ 
        events.shape: (4551, 4)
        labels.shape: (4551,)
        """

        testset_x, testset_y = [], []
        idx_order = []
        idx = 0
        
        for idx, (event, label) in enumerate(zip(events, labels)):
            # if use_prob_embedding:
            #     event = self.probability_based_embedding(event, rv)            
            # # print(f"{idx}th event {event}: label {label}")
            
            testset_x.append(event)
            testset_y.append([label])
            idx_order.append(idx)
                    
        testset_x, testset_y, truncated_idxs = self.truncate(testset_x, testset_y, idx_order, slen=slen)
        # print("testset_x.shape:", testset_x.shape)
        # print("testset_y.shape:", testset_y.shape)
        """ 
        testset_x.shape: (4542, 10, 4)
        testset_y.shape: (4542, 10, 1)
        """
        y_pred = self.model.predict(testset_x)
        # print("y_pred.shape:",y_pred.shape)
        # print("y_pred[:1]:",y_pred[:1])
        
        """ 
        y_pred.shape: (4542, 10, 1)
        """




        '''acumulates predictions'''
        idx = 0
        predictions = {}   
        for pred in y_pred: # 遍历测试集logit

            '''iterates through remaining truncated seq len if surpassing the limit'''
            if len(y_pred) - idx > slen:
                lst = range(len(pred))
            else:
                lst = range(len(y_pred)-idx)

            '''acumulates truncated predictions: e.g. {idx1: [1,1,0,0], idx2: [1,0,0,1], idx3: [0,0,1,1], ...]'''
            for i in lst:
                if idx + i not in predictions:
                    predictions[idx + i] = []
                predictions[idx + i].append(pred[i][0])
            idx += 1
            
        '''looks like it takes the average of predictions for each truncated sequence? not sure'''
        results = []
        for idx in range(len(testset_x) - slen + 1):
            res = sum(predictions[idx])/len(predictions[idx])
            results.append(res)
            
        ret_probs = []
        ret_idxs = []
        
        for idx in range(len(results)):
            if results[idx] > 0.01:
                prob = results[idx]
                ret_probs.append(prob)
                ret_idxs.append(idx)


        ret_probs = np.array(ret_probs)
        ret_idxs = np.array(ret_idxs)
                
        return ret_probs, ret_idxs

    def def_model(self, input_length, output_length, input_dim=4, output_dim=1, hidden_units=128):
        # print('define seq2seq model architecture')    
        
        # # 使用示例
        # input_shape = (input_length, input_dim)  # 输入形状
        # output_shape = (output_length, output_dim)  # 输出形状
        # hidden_units = 128  # 隐藏层单元数

        # model = seq2seq_model(input_shape, output_shape, hidden_units)
        # self.model = model      
        print("--------------------create seq2seq------------------------")        
        # train_input = Input(shape=(10, 4))
        # train_output = Input(shape=(10, 1))        
        train_input = Input(shape=(input_length, input_dim))
        train_output = Input(shape=(output_length, output_dim))        
                    
        print("train_input.shape:", train_input.shape)
        print("train_output.shape:", train_output.shape)
        """ 
        train_input.shape: (None, 10, 4)
        train_output.shape: (None, 10, 1)
        """ 
        
        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
                units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
                return_sequences=True, return_state=True)(train_input)

        encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
        encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

        decoder_input = RepeatVector(train_output.shape[1])(encoder_last_h)
        decoder_stack_h = LSTM(units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
                return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        
        
        print("encoder_stack_h.shape:",encoder_stack_h.shape)
        print("decoder_stack_h.shape:",decoder_stack_h.shape)
        
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2,2])
        attention = Activation('softmax')(attention)

        context = dot([attention, encoder_stack_h], axes=[2,1])
        context = BatchNormalization(momentum=0.6)(context)

        decoder_combined_context = concatenate([context, decoder_stack_h])
        out = TimeDistributed(Dense(train_output.shape[2]))(decoder_combined_context)
        out = Activation('sigmoid')(out)

        print("train_input.shape:",train_input.shape)                
        print("out.shape:",out.shape)       
        
        """ 
        encoder_stack_h.shape: (None, 10, 128)
        decoder_stack_h.shape: (None, 10, 128)
        train_input.shape: (None, 10, 4)
        out.shape: (None, 10, 1)
        """ 
        
        self.model = Model(inputs=train_input, outputs=out)
        print("--------------------end create seq2seq------------------------")    
                                
    def stdtrain(self, events, labels, exp_result_dir):

        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        print("events.shape:",events.shape)
        print("labels.shape:",labels.shape)
        
        slen = self.args.sequence_length
        print("self.args.sequence_length:",self.args.sequence_length)

        trainset_x, trainset_y = [], []
        idx_order = []
        idx = 0
        
        for idx, (event, label) in enumerate(zip(events, labels)):   
            # print("idx:",idx)         
            trainset_x.append(event)
            trainset_y.append([label])
            idx_order.append(idx)
            
        trainset_x, trainset_y, truncated_idxs = self.truncate(trainset_x, trainset_y, idx_order, slen=slen)
        trainset_x, trainset_y = shuffle(trainset_x, trainset_y)

        print("trainset_x.shape:",trainset_x.shape)
        print("trainset_y.shape:",trainset_y.shape)
        
        """
        trainset_x.shape: (19809, 10, 4)
        trainset_y.shape: (19809, 10, 1)
        """

        # 配置模型的训练过程
        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=self.args.lr), metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=self.args.patience, verbose=1)    
        timer_callback = EpochTimer()
        callbacks = [early_stop, timer_callback]
        # tf.compat.v1.experimental.output_all_intermediates(True)
        history = self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.seq2seq_batchsize, epochs=self.args.seq2seq_epochs, verbose=2, callbacks=callbacks, validation_split=0.2)
          
        # raise Exception("maggie stop")
          
        # maggie
        epo_train_loss = history.history['loss']
        epo_val_loss = history.history['val_loss']
        epo_train_acc = history.history['accuracy']
        epo_val_acc = history.history['val_accuracy']

        # 将准确率历史记录转换为百分比
        epo_train_acc = [accuracy * 100 for accuracy in epo_train_acc]
        epo_val_acc = [accuracy * 100 for accuracy in epo_val_acc]
        
        #--------save plt---------            
        loss_png_name = f'Loss of standard trained {self.modelname}'
        accuracy_png_name = f'Accuracy of standard trained {self.modelname}'        
                   
        plt.plot(list(range(len(epo_train_loss))), epo_train_loss, label='Train Loss', marker='o')
        plt.plot(list(range(len(epo_val_loss))), epo_val_loss, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{loss_png_name}')
        plt.show()
        plt.savefig(f'{exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(list(range(len(epo_train_acc))), epo_train_acc, label='Train Accuracy', marker='o')
        plt.plot(list(range(len(epo_val_acc))), epo_val_acc, label='Validation Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        # plt.ylim(0, 1)
        plt.legend(loc='best',frameon=True)
        plt.title(f'{accuracy_png_name}')        
        plt.show()
        plt.savefig(f'{exp_result_dir}/{accuracy_png_name}.png')
        plt.close()

    def save_model(self, save_path):
        self.model.save(save_path)

    def load_model(self, model_path):
        from keras.models import load_model
        self.model = load_model(model_path)
 
    def retrain(self, events, labels, exp_result_dir):
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        print("events.shape:",events.shape)
        print("labels.shape:",labels.shape)
        """ 
        events.shape: (19818, 4)
        labels.shape: (19818,)
        """
        
        slen = self.args.sequence_length
        print("self.args.sequence_length:",self.args.sequence_length)

        trainset_x, trainset_y = [], []
        idx_order = []
        idx = 0
        
        for idx, (event, label) in enumerate(zip(events, labels)):   
            # print("idx:",idx)         
            trainset_x.append(event)
            trainset_y.append([label])
            idx_order.append(idx)
            
        trainset_x, trainset_y, truncated_idxs = self.truncate(trainset_x, trainset_y, idx_order, slen=slen)
        trainset_x, trainset_y = shuffle(trainset_x, trainset_y)

        print("trainset_x.shape:",trainset_x.shape)
        print("trainset_y.shape:",trainset_y.shape)
        
        """
        trainset_x.shape: (19809, 10, 4)
        trainset_y.shape: (19809, 10, 1)
        """

        # 配置模型的训练过程
        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=self.args.lr), metrics=['accuracy'])
        print("compile success")
        early_stop = EarlyStopping(monitor='val_loss', patience=self.args.patience, verbose=1)    
        timer_callback = EpochTimer()
        callbacks = [early_stop, timer_callback]
        # tf.compat.v1.experimental.output_all_intermediates(True)
        # 设置训练模式
        # 切换为训练模式
        self.model.trainable = True
        print("trainable success")

        # tf.keras.backend.set_learning_phase(1)  # 或者使用 tf.constant(1)
        # history = self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.seq2seq_batchsize, epochs=self.args.seq2seq_epochs, verbose=2, callbacks=callbacks, validation_split=0.2,training=True)
        history = self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.seq2seq_batchsize, epochs=self.args.seq2seq_epochs, verbose=2, callbacks=callbacks, validation_split=0.2)

        # tf.keras.backend.set_learning_phase(0)  # 或者使用 tf.constant(0)  
        # raise Exception("maggie stop")
          
        # maggie
        epo_train_loss = history.history['loss']
        epo_val_loss = history.history['val_loss']
        epo_train_acc = history.history['accuracy']
        epo_val_acc = history.history['val_accuracy']

        # 将准确率历史记录转换为百分比
        epo_train_acc = [accuracy * 100 for accuracy in epo_train_acc]
        epo_val_acc = [accuracy * 100 for accuracy in epo_val_acc]
        
        #--------save plt---------            
        loss_png_name = f'Loss of retrained {self.modelname}'
        accuracy_png_name = f'Accuracy of retrained {self.modelname}'        
                   
        plt.plot(list(range(len(epo_train_loss))), epo_train_loss, label='Train Loss', marker='o')
        plt.plot(list(range(len(epo_val_loss))), epo_val_loss, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{loss_png_name}')
        plt.show()
        plt.savefig(f'{exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(list(range(len(epo_train_acc))), epo_train_acc, label='Train Accuracy', marker='o')
        plt.plot(list(range(len(epo_val_acc))), epo_val_acc, label='Validation Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        # plt.ylim(0, 1)
        plt.legend(loc='best',frameon=True)
        plt.title(f'{accuracy_png_name}')        
        plt.show()
        plt.savefig(f'{exp_result_dir}/{accuracy_png_name}.png')
        plt.close()        