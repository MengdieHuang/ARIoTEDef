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

    # def predict(self, dataset):     
    #     pred = self.model.predict(dataset)
    #     return pred
            