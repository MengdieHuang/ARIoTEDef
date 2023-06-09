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
            
        print(f"{self.modelname} trainset_min:{self.trainset_min}")
        print(f"{self.modelname} trainset_max:{self.trainset_max}")
        print(f"{self.modelname} testset_min:{self.testset_min}")
        print(f"{self.modelname} testset_max:{self.testset_max}")            
                
    def def_model(self, input_dim=41, output_dim=1, timesteps=1):  

        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")  # 指定使用的 GPU 设备
        
        with strategy.scope():
            
            model = Sequential()
            # model.add(LSTM(units=128, activation='relu', return_sequences=True, input_shape=(timesteps, input_dim)))
            # 输出128维
            # model.add(LSTM(units=128, activation='relu', return_sequences=True, input_dim=input_dim))
            model.add(LSTM(units=128, activation='relu', return_sequences=True, input_shape=(timesteps, int(input_dim / timesteps))))
            
            
            model.add(LSTM(units=128, activation='relu', return_sequences=True))                        
            # 输出128维
            model.add(Dense(units=output_dim, activation='sigmoid'))
            # 输出1维
            model.add(Flatten())
        
            # metrics = [tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall()]
            metrics = ['accuracy']

            # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
            
            # model.summary()
            self.model = model

    def stdtrain(self, timesteps, exp_result_dir):
        
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print(" cuda and GPU are available")

        print(f"prepare training set and test set for learning {self.modelname} ")
        print("shuffle training set")       
        trainset_x = self.dataset['train'][0]
        trainset_y = self.dataset['train'][1]
        # testset_x = self.dataset['test'][0]
        # testset_y = self.dataset['test'][1]    
        print("trainset_x.shape:",trainset_x.shape)
        print("trainset_y.shape:",trainset_y.shape)
        """ 
        trainset_x.shape: (19152, 41)
        trainset_y.shape: (19152,)
        """
        # print("testset_x.shape:",testset_x.shape)
        # print("testset_y.shape:",testset_y.shape)

        trainset_x, trainset_y = shuffle(trainset_x, trainset_y)
        print("trainset_x.shape:",trainset_x.shape)
        print("trainset_y.shape:",trainset_y.shape)
        """ 
        trainset_x.shape: (19152, 41)
        trainset_y.shape: (19152,)
        """
        
        # trainset_x = trainset_x.reshape((trainset_x.shape[0], 1, trainset_x.shape[1]))
        trainset_x = trainset_x.reshape((trainset_x.shape[0], timesteps, int(math.ceil(trainset_x.shape[1] / timesteps))))

        print("trainset_x.shape:",trainset_x.shape)
        """
        trainset_x.shape: (19152, 1, 41)
        """

        trainset_x, valset_x, trainset_y, valset_y = train_test_split(trainset_x, trainset_y, test_size=0.1, random_state=42)
        print("trainset_x.shape:",trainset_x.shape)
        print("trainset_y.shape:",trainset_y.shape)        
        print("valset_x.shape:",valset_x.shape)
        print("valset_y.shape:",valset_y.shape)        
        """ 
        trainset_x.shape: (17236, 1, 41)
        trainset_y.shape: (17236,)
        valset_x.shape: (1916, 1, 41)
        valset_y.shape: (1916,)
        """

        new_train_size = (len(trainset_x) // 32) * 32
        trainset_x = trainset_x[:new_train_size]
        trainset_y = trainset_y[:new_train_size]
        print("trainset_x.shape:",trainset_x.shape)
        print("trainset_y.shape:",trainset_y.shape)
        
        new_val_size = (len(valset_x) // 32) * 32
        valset_x = valset_x[:new_val_size]
        valset_y = valset_y[:new_val_size]
        print("valset_x.shape:",valset_x.shape)
        print("valset_y.shape:",valset_y.shape)    
        
                        
        """  
        trainset_x.shape: (17216, 1, 41)
        trainset_y.shape: (17216,)
        valset_x.shape: (1888, 1, 41)
        valset_y.shape: (1888,)
        """
      
        early_stop = EarlyStopping(monitor='val_loss', patience=self.args.patience, verbose=1) 
        # history=self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.batchsize, epochs=self.args.ps_epochs, verbose=2, callbacks=[early_stop], validation_data=(valset_x,valset_y))       
      
      
        timer_callback = EpochTimer()
        callbacks = [early_stop, timer_callback]
        history=self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.batchsize, epochs=self.args.ps_epochs, verbose=2, callbacks=callbacks, validation_data=(valset_x,valset_y))       

        epo_train_loss = history.history['loss']
        epo_val_loss = history.history['val_loss']
        epo_train_acc = history.history['accuracy']
        epo_val_acc = history.history['val_accuracy']
        epo_cost_time = timer_callback.epoch_times

        #--------save plt---------            
        loss_png_name = f'Loss of standard trained {self.modelname}'
        accuracy_png_name = f'Accuracy of standard trained {self.modelname}'        
        time_png_name = f'Cost time of standard trained {self.modelname}'
                   
        plt.plot(epo_train_loss, label='Train Loss')
        plt.plot(epo_val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{loss_png_name}')
        plt.show()
        plt.savefig(f'{exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(epo_train_acc, label='Train Accuracy')
        plt.plot(epo_val_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'{accuracy_png_name}')        
        plt.show()
        plt.savefig(f'{exp_result_dir}/{accuracy_png_name}.png')
        plt.close()

        plt.plot(epo_cost_time)
        plt.xlabel('Epoch')
        plt.ylabel('Cost Time')
        plt.legend()
        plt.title(f'{time_png_name}')        
        plt.show()
        plt.savefig(f'{exp_result_dir}/{time_png_name}.png')
        plt.close()

        # batch_size = self.args.batchsize
        # num_batches = len(trainset_x) // batch_size
        
        # print(f"batch_size={self.args.batchsize}, epoch_num={self.args.ps_epochs}")
        # # batch_size=32, epoch_num=50

        # # 初始化变量用于记录训练过程中的指标
        # train_losses = []
        # val_losses = []
        # train_accuracies = []
        # val_accuracies = []
        
        # # 训练循环
        # for epoch in range(self.args.ps_epochs):
        #     print(f"Epoch {epoch + 1}/{self.args.ps_epochs}")

        #     train_loss_epoch = 0
        #     train_acc_epoch = 0
                
        #     # 逐批次进行训练
        #     for batch_idx in range(num_batches):
        #         batch_start = batch_idx * batch_size
        #         batch_end = (batch_idx + 1) * batch_size

        #         batch_x = trainset_x[batch_start:batch_end]
        #         batch_y = trainset_y[batch_start:batch_end]

        #         # # 使用 train_on_batch() 函数训练模型
        #         # loss, accuracy = self.model.train_on_batch(batch_x, batch_y)
        #         # # result = self.model.train_on_batch(batch_x, batch_y)

        #         # # 设置 eager learning phase
        #         # with tf.GradientTape():
        #         #     tf.keras.backend.set_learning_phase(1)
        #         loss, accuracy = self.model.train_on_batch(batch_x, batch_y)                

        
        #         print("loss:",loss)
        #         print("accuracy:",accuracy)
                
        #         train_loss_epoch += loss
        #         train_acc_epoch += accuracy
        
        #         print(f"Batch {batch_idx + 1}/{num_batches} - loss: {loss:.4f} - accuracy: {accuracy:.4f}")
        #         raise Exception("maggie stop here!!!!")

        #     train_loss_epoch /= num_batches
        #     train_acc_epoch /= num_batches

        #     # 在每个 epoch 结束后，计算验证集上的损失并检查早停条件
        #     # val_loss, _ = self.model.evaluate(valset_x, valset_y, verbose=0)
            
        #     # 计算验证集上的损失和准确率
        #     val_loss, val_acc = self.model.evaluate(valset_x, valset_y, verbose=0)

        #     print(f"Train Loss: {train_loss_epoch:.4f} - Train Accuracy: {train_acc_epoch:.4f}")
        #     print(f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}")
            
        #     # 保存指标
        #     train_losses.append(train_loss_epoch)
        #     val_losses.append(val_loss)
        #     train_accuracies.append(train_acc_epoch)
        #     val_accuracies.append(val_acc)
            
        #     # 检查早停条件
        #     if early_stop and early_stop.on_epoch_end(epoch, logs={'val_loss': val_loss, 'val_acc': val_acc}):
        #         print("Early stopping triggered.")
        #         break            
             
    def evaluate(self, testset_x, testset_y):
        
        # 使用model.evaluate计算测试损失
        # test_los = self.model.evaluate(testset_x, testset_y)
        test_los, _ = self.model.evaluate(testset_x, testset_y)
        

        output = self.model.predict(testset_x)
        # print("output:",output)
        # print("output.shape:",output.shape)
        
        """ 
        output: 
        [[5.2461257e-15]
        [1.0004375e-10]
        [1.0007104e-10]
        ...
        [1.0000000e+00]
        [1.0000000e+00]
        [1.0000000e+00]]
        output.shape: (4224, 1)
        """

        
        # # 检查目标值的不同类别数量
        # num_classes = len(np.unique(testset_y))
        # print("Number of testset_y classes:", num_classes)

        # num_classes = len(np.unique(output))
        # print("Number of output classes:", num_classes)
        
        # """ 
        # Number of testset_y classes: 2
        # Number of output classes: 400
        # """
        
        predicts = []
        for p in output:
            ret = (p[0] > 0.5).astype("int32")
            # print(f"direct calculation number:{p[0]}, predict label:{ret}")
            predicts.append(ret)
            
        output = np.array(predicts)
        
        # num_classes = len(np.unique(output))
        # print("Number of output classes:", num_classes)
        # print("output.shape:",output.shape)     
        # """ 
        # Number of output classes: 2
        # output.shape: (4224,)
        # """
        
        test_TN, test_FP, test_FN, test_TP = confusion_matrix(testset_y, output).ravel()
        test_acc = accuracy_score(testset_y, output)
        test_recall = recall_score(testset_y, output, average='macro')
        test_precision = precision_score(testset_y, output, average='macro')
        test_F1 = f1_score(testset_y, output, average='macro')
        test_FPR = test_FP / (test_FP + test_TN)
        
        return test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1
    
    def test(self, testset_x, testset_y, timesteps, exp_result_dir):
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print(" cuda and GPU are available")

        print(f"prepare test set for evaluating {self.modelname} ")
        # testset_x = self.dataset['test'][0]
        # testset_y = self.dataset['test'][1]    
        print("testset_x.shape:",testset_x.shape)
        print("testset_y.shape:",testset_y.shape)
        
        testset_x = testset_x.reshape((testset_x.shape[0], timesteps, int(math.ceil(testset_x.shape[1] / timesteps))))
        print("testset_x.shape:",testset_x.shape)
        
        """ 
        testset_x.shape: (4233, 41)
        testset_y.shape: (4233,)
        testset_x.shape: (4233, 1, 41)
        """
    
        # new_test_size = (len(testset_x) // 32) * 32
        # testset_x = testset_x[:new_test_size]
        # testset_y = testset_y[:new_test_size]
        # print("testset_x.shape:",testset_x.shape)
        # print("testset_y.shape:",testset_y.shape)    
        # """ 
        # testset_x.shape: (4224, 1, 41)
        # testset_y.shape: (4224,)
        # """        
        # raise Exception("maggie")
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1 = self.evaluate(testset_x, testset_y)
        
        
        return test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1
             
    def generate_advmail(self,timesteps, exp_result_dir):        
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print(" cuda and GPU are available")

        print(f"prepare test set for generating adversarial testset against {self.modelname} ")
        cle_testset_x = self.dataset['test'][0]
        cle_testset_y = self.dataset['test'][1]    
        print("cle_testset_x.shape:",cle_testset_x.shape)
        print("cle_testset_y.shape:",cle_testset_y.shape)
        print("cle_testset_y[:10]:",cle_testset_y[:10])        
        """
        cle_testset_x.shape: (4233, 41)
        cle_testset_y.shape: (4233,)
        cle_testset_y[:10]: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        """
        print(f"{self.modelname} cle_testset_min:{self.testset_min}")
        print(f"{self.modelname} cle_testset_max:{self.testset_max}")      
       
        """ 
        attack-detector cle_testset_min:-3.30513966135273
        attack-detector cle_testset_max:21.144568401380717
        """
        # benign 0 malicious 1
        
        
        
        # extract malicious set
        condition = cle_testset_y.astype(bool)
        print("condition.shape:",condition.shape)
        print("condition[:10]:",condition[:10])
        
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
        
        # malicious_cle_testset_x = np.extract(condition,cle_testset_x)
        cond=np.expand_dims(condition,1)
        print("cond.shape:",cond.shape)
        # 创建形状为(4233, 41)的全False数组
        cond_expend = np.full((cle_testset_x.shape[0], cle_testset_x.shape[1]), False, dtype=bool)
        # 将条件数组广播到result数组中
        cond = np.logical_or(cond_expend, cond)        
        print("cond.shape:",cond.shape)        
        """
        cond.shape: (4233, 1)
        cond.shape: (4233, 41)
        """
        
        malicious_cle_testset_x = np.extract(cond,cle_testset_x)
        print("malicious_cle_testset_x.shape:",malicious_cle_testset_x.shape)        

        """
        malicious_cle_testset_x.shape: (5043,)

        """        
        malicious_cle_testset_x = np.reshape(malicious_cle_testset_x, (malicious_cle_testset_y.shape[0], cle_testset_x.shape[1]))        
        print("malicious_cle_testset_x.shape:",malicious_cle_testset_x.shape)
        """ 
        malicious_cle_testset_x.shape: (123, 41)
        """
        
        
        malicious_cle_testset_x = malicious_cle_testset_x.reshape((malicious_cle_testset_x.shape[0], timesteps, int(math.ceil(malicious_cle_testset_x.shape[1] / timesteps))))
        print("malicious_cle_testset_x.shape:",malicious_cle_testset_x.shape)
        
        """ 
        malicious_cle_testset_x.shape: (123, 1, 41)
        """
        
        new_test_size = (len(malicious_cle_testset_x) // 32) * 32
        malicious_cle_testset_x = malicious_cle_testset_x[:new_test_size]
        malicious_cle_testset_y = malicious_cle_testset_y[:new_test_size]
        print("malicious_cle_testset_x.shape:",malicious_cle_testset_x.shape)
        print("malicious_cle_testset_y.shape:",malicious_cle_testset_y.shape)    
        """ 
        malicious_cle_testset_x.shape: (96, 1, 41)
        malicious_cle_testset_y.shape: (96,)
        """             

        print("self.testset_min:",self.testset_min)
        print("self.testset_max:",self.testset_max)
        
        print("self.args.eps:",self.args.eps)
        print("self.args.eps_step:",self.args.eps_step)
        print("self.args.max_iter:",self.args.max_iter)
        
        import sys
        sys.stdout.flush()
        print("正在导入art库...", flush=True)
        from art.estimators.classification.keras import KerasClassifier
        from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
        print("art库导入完成!", flush=True)   
             
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
            
        print(f"{self.modelname} trainset_min:{self.trainset_min}")
        print(f"{self.modelname} trainset_max:{self.trainset_max}")
        print(f"{self.modelname} testset_min:{self.testset_min}")
        print(f"{self.modelname} testset_max:{self.testset_max}")  
    
    # def def_model(self):
    #     from keras.layers import RepeatVector, TimeDistributed, BatchNormalization, Activation, Input, dot, concatenate
    #     from tensorflow.keras.optimizers import Adam
    #     from keras.models import Model
        
        
    #     print(f"define {self.modelname} ")
    #     slen = self.args.sequence_length
    #     # rv = self.args.rv
    #     # permute_truncated = self.args.permute_truncated
    #     # use_prob_embedding = self.args.use_prob_embedding
    #     print("self.args.sequence_length:",self.args.sequence_length)
    #     # print("self.args.rv:",self.args.rv)
    #     # print("self.args.permute_truncated:",self.args.permute_truncated)
    #     # print("self.args.use_prob_embedding:",self.args.use_prob_embedding)

    #     """ 
    #     training infection-seq2seq
    #     self.args.sequence_length: 10
    #     self.args.rv: 1
    #     self.args.permute_truncated: False
    #     self.args.use_prob_embedding: False
    #     """
        
    #     print("prepare training set")


    #     in_, out_ = [], []
    #     idx_order = []
    #     idx = 0
        
    #     for idx, (event, label) in enumerate(zip(events, labels)):
    #         # if use_prob_embedding:
    #         #     event = self.probability_based_embedding(event, rv)            
    #         # print(f"{idx}th event {event}: label {label}")
            
    #         in_.append(event)
    #         out_.append([label])
    #         idx_order.append(idx)

    #     X_in, X_out, truncated_idxs = self.truncate(in_, out_, idx_order, slen=slen)
    #     X_out_labels = np.array(X_out)[:,:,0].tolist()

    #     # if permute_truncated:
    #     #     print("Permute Truncated is enabled")
    #     #     X_in, perm_truncated_idxs = self.permute_truncated(X_in, X_out_labels, truncated_idxs, slen=slen, inplace=False)
    #     # else:
    #     #     print("Permute Truncated is disabled")

    #     # if use_prob_embedding:
    #     #     X_in = np.expand_dims(X_in, axis=-1)
        
    #     print("input_train.shape:",input_train.shape)
    #     print("output_train.shape:",output_train.shape)
    #     input_train = Input(shape=(X_in.shape[-2], X_in.shape[-1]))
    #     output_train = Input(shape=(X_out.shape[-2], X_out.shape[-1]))
        
    #     print("input_train.shape:",input_train.shape)
    #     print("output_train.shape:",output_train.shape)
                
    #     print("create seq2seq model")

        
    #     encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
    #             units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
    #             return_sequences=True, return_state=True)(input_train)

    #     encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
    #     encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

    #     decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
    #     decoder_stack_h = LSTM(units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
    #             return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        
    #     attention = dot([decoder_stack_h, encoder_stack_h], axes=[2,2])
    #     attention = Activation('softmax')(attention)

    #     context = dot([attention, encoder_stack_h], axes=[2,1])
    #     context = BatchNormalization(momentum=0.6)(context)

    #     decoder_combined_context = concatenate([context, decoder_stack_h])
    #     out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
    #     out = Activation('sigmoid')(out)

    #     self.model = Model(inputs=input_train, outputs=out)
        
    #     opt = Adam(learning_rate=0.01, clipnorm=1)
    #     binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
    #                             from_logits=False,
    #                             label_smoothing=0.0,
    #                             reduction="auto",
    #                             name="binary_crossentropy",
    #                         )

    #     metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    #     self.model.compile(loss=binary_crossentropy, optimizer=opt, metrics=metrics)        


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
                
            permute_idxs = permute_idxs + list(np.random.permutation(repeating_seq))
            if i < 20 and enable_permute_prints:
                print("permuting {} with idxs {}".format(x_seq_in, permute_idxs))
                print("permuting {} with idxs {}".format(seq_idxs, permute_idxs))
            self.permute_sublist_inplace(x_seq_in, permute_idxs)    
            self.permute_sublist_inplace(seq_idxs, permute_idxs)
            #print(seq_idxs)
        if not inplace:
            return X_in, truncated_idxs
    
    def stdtrain(self, events, labels, exp_result_dir, permute_truncated=True):
        from keras.layers import RepeatVector, TimeDistributed, BatchNormalization, Activation, Input, dot, concatenate
        from tensorflow.keras.optimizers import Adam
        from keras.models import Model    
        
        print(f"training {self.modelname}")
        
        slen = self.args.sequence_length
        rv = self.args.rv
        permute_truncated = self.args.permute_truncated
        use_prob_embedding = self.args.use_prob_embedding
        print("self.args.sequence_length:",self.args.sequence_length)
        print("self.args.rv:",self.args.rv)
        print("self.args.permute_truncated:",self.args.permute_truncated)
        print("self.args.use_prob_embedding:",self.args.use_prob_embedding)

        """ 
        training infection-seq2seq
        self.args.sequence_length: 10
        self.args.rv: 1
        self.args.permute_truncated: False
        self.args.use_prob_embedding: False
        """
        
        print("prepare training set")


        in_, out_ = [], []
        idx_order = []
        idx = 0
        
        for idx, (event, label) in enumerate(zip(events, labels)):
            if use_prob_embedding:
                event = self.probability_based_embedding(event, rv)            
            # print(f"{idx}th event {event}: label {label}")
            
            in_.append(event)
            out_.append([label])
            idx_order.append(idx)

        X_in, X_out, truncated_idxs = self.truncate(in_, out_, idx_order, slen=slen)
        X_out_labels = np.array(X_out)[:,:,0].tolist()

        if permute_truncated:
            print("Permute Truncated is enabled")
            X_in, perm_truncated_idxs = self.permute_truncated(X_in, X_out_labels, truncated_idxs, slen=slen, inplace=False)
        else:
            print("Permute Truncated is disabled")

        if use_prob_embedding:
            X_in = np.expand_dims(X_in, axis=-1)

        input_train = Input(shape=(X_in.shape[-2], X_in.shape[-1]))
        output_train = Input(shape=(X_out.shape[-2], X_out.shape[-1]))


        print("create seq2seq")
        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
                units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
                return_sequences=True, return_state=True)(input_train)

        encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
        encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

        decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
        decoder_stack_h = LSTM(units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
                return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2,2])
        attention = Activation('softmax')(attention)

        context = dot([attention, encoder_stack_h], axes=[2,1])
        context = BatchNormalization(momentum=0.6)(context)

        decoder_combined_context = concatenate([context, decoder_stack_h])
        out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
        out = Activation('sigmoid')(out)

        self.model = Model(inputs=input_train, outputs=out)
        
        opt = Adam(learning_rate=0.01, clipnorm=1)
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
                                from_logits=False,
                                label_smoothing=0.0,
                                reduction="auto",
                                name="binary_crossentropy",
                            )

        metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        self.model.compile(loss=binary_crossentropy, optimizer=opt, metrics=metrics)

        # 以上模型架构的定义能不能独立出去
        X_in, X_out = shuffle(X_in,X_out)
        
        print("X_in.shape:", X_in.shape)
        print("X_out.shape:", X_out.shape)

        print("X_out[:, :, :1].shape:",X_out[:, :, :1].shape)
        """ 
        X_in.shape: (19799, 10, 4)
        X_out.shape: (19799, 10, 1)
        X_out[:, :, :1].shape: (19799, 10, 1)
        从张量 X_out 中选择所有的行（:表示选择所有行）、所有的列（:表示选择所有列），并且只选择第一个维度上的第一个元素（:1表示选择索引为0的元素）
        """
        
        history = self.model.fit(x=X_in, y=X_out[:, :, :1], validation_split=0.2, epochs=self.args.seq2seq_epochs, verbose=2, batch_size=100)
  
        # maggie
        epo_train_loss = history.history['loss']
        epo_val_loss = history.history['val_loss']
        epo_train_acc = history.history['accuracy']
        epo_val_acc = history.history['val_accuracy']

        #--------save plt---------            
        loss_png_name = f'Loss of standard trained {self.modelname}'
        accuracy_png_name = f'Accuracy of standard trained {self.modelname}'        
                   
        plt.plot(epo_train_loss, label='Train Loss')
        plt.plot(epo_val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{loss_png_name}')
        plt.show()
        plt.savefig(f'{exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(epo_train_acc, label='Train Accuracy')
        plt.plot(epo_val_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'{accuracy_png_name}')        
        plt.show()
        plt.savefig(f'{exp_result_dir}/{accuracy_png_name}.png')
        plt.close()

    def evaluate(self, testset_x, testset_y):
        
        
        # 使用model.evaluate计算测试损失
        # test_los = self.model.evaluate(testset_x, testset_y)
        test_los, _ = self.model.evaluate(testset_x, testset_y)
        

        output = self.model.predict(testset_x)
        print("output:",output)
        print("output.shape:",output.shape)
      
        raise Exception("maggie")
        
        predicts = []
        for p in output:
            ret = (p[0] > 0.5).astype("int32")
            # print(f"direct calculation number:{p[0]}, predict label:{ret}")
            predicts.append(ret)
            
        output = np.array(predicts)
        
       
        
        test_TN, test_FP, test_FN, test_TP = confusion_matrix(testset_y, output).ravel()
        test_acc = accuracy_score(testset_y, output)
        test_recall = recall_score(testset_y, output, average='macro')
        test_precision = precision_score(testset_y, output, average='macro')
        test_F1 = f1_score(testset_y, output, average='macro')
        test_FPR = test_FP / (test_FP + test_TN)
        
        return test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1
    
    
    def test(self, testset_x, testset_y, exp_result_dir):
            
        # print("events.shape:",events.shape)
        # print("labels.shape:",labels.shape)

        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print(" cuda and GPU are available")

        print(f"prepare test set for evaluating {self.modelname} ")
        # testset_x = self.dataset['test'][0]
        # testset_y = self.dataset['test'][1]    
        print("testset_x.shape:",testset_x.shape)
        print("testset_y.shape:",testset_y.shape)
        """ 
        testset_x.shape: (4224, 4)
        testset_y.shape: (4224,)
        """
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1 = self.evaluate(testset_x, testset_y)
        
        
        return test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_FPR, test_F1
            