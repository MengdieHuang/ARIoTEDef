
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def SS_normalizedata(dataset_x):
    # dataset_min = np.min(dataset_x)
    # dataset_max = np.max(dataset_x)
    # print(f"dataset_min:{dataset_min}")
    # print(f"dataset_max:{dataset_max}") 
    
    # scaler = StandardScaler().fit(dataset_x)
    # dataset_x = scaler.transform(dataset_x)
    scaler = StandardScaler()
    dataset_x = scaler.fit_transform(dataset_x)
    
    # dataset_min = np.min(dataset_x)
    # dataset_max = np.max(dataset_x)
    # print(f"dataset_min:{dataset_min}")
    # print(f"dataset_max:{dataset_max}") 
          
    return dataset_x

def MinMax_normalizedata(dataset_x):
    # dataset_min = np.min(dataset_x)
    # dataset_max = np.max(dataset_x)
    # print(f"dataset_min:{dataset_min}")
    # print(f"dataset_max:{dataset_max}") 
    

    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()
    # 拟合并转换数据
    dataset_x = scaler.fit_transform(dataset_x)


    # dataset_min = np.min(dataset_x)
    # dataset_max = np.max(dataset_x)
    # print(f"dataset_min:{dataset_min}")
    # print(f"dataset_max:{dataset_max}") 
          
    return dataset_x

def normalize_multistep_dataset(multistep_dataset):

    norm_train_data_infection = SS_normalizedata(multistep_dataset['infection']['train'][0]) 
    train_label_infection = multistep_dataset['infection']['train'][1]
    
    norm_test_data_infection = SS_normalizedata(multistep_dataset['infection']['test'][0]) 
    test_label_infection = multistep_dataset['infection']['test'][1]
    
    norm_train_data_attack = SS_normalizedata(multistep_dataset['attack']['train'][0]) 
    train_label_attack = multistep_dataset['attack']['train'][1]
    
    norm_test_data_attack = SS_normalizedata(multistep_dataset['attack']['test'][0]) 
    test_label_attack = multistep_dataset['attack']['test'][1]
    
    norm_train_data_reconnaissance = SS_normalizedata(multistep_dataset['reconnaissance']['train'][0]) 
    train_label_reconnaissance = multistep_dataset['reconnaissance']['train'][1]
    
    norm_test_data_reconnaissance = SS_normalizedata(multistep_dataset['reconnaissance']['test'][0])     
    test_label_reconnaissance = multistep_dataset['reconnaissance']['test'][1]
    
    norm_multistep_dataset = {"infection": 
                            {
                            'train': [norm_train_data_infection, train_label_infection], 
                            'test': [norm_test_data_infection, test_label_infection]
                            },
                "attack": 
                            {
                            'train': [norm_train_data_attack, train_label_attack], 
                            'test': [norm_test_data_attack, test_label_attack]
                            },
                "reconnaissance": 
                            {
                            'train': [norm_train_data_reconnaissance, train_label_reconnaissance], 
                            'test': [norm_test_data_reconnaissance, test_label_reconnaissance]
                            }
                }        
                
    return norm_multistep_dataset