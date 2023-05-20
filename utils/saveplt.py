"""
Author: maggie
Date:   2021-06-15
@copyright
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def SaveTotalTimeCurve(model,dataset,exp_result_dir,global_cost_time,png_name):
    x = list(range(len(global_cost_time)))
    y = global_cost_time
    # test_x = list(range(len(global_test_acc)))
    # test_y = global_test_acc
    plt.title(f'{png_name}')
    plt.plot(x, y, color='black') 
    # plt.plot(test_x, test_y, color='blue', label='on testset accuracy') 
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Total Cost Time (seconds)')
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()
    
def SaveTimeCurve(model,dataset,exp_result_dir,global_cost_time,png_name):
    x = list(range(len(global_cost_time)))
    y = global_cost_time
    # test_x = list(range(len(global_test_acc)))
    # test_y = global_test_acc
    plt.title(f'{png_name}')
    plt.plot(x, y, color='black') 
    # plt.plot(test_x, test_y, color='blue', label='on testset accuracy') 
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Cost Time (seconds)')
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()
    
def SaveAccuracyCurve(model,dataset,exp_result_dir,global_train_acc,global_test_acc,png_name):
    train_x = list(range(len(global_train_acc)))
    train_y = global_train_acc
    test_x = list(range(len(global_test_acc)))
    test_y = global_test_acc
    plt.title(f'{png_name}')
    plt.plot(train_x, train_y, color='black', label='train acc') 
    plt.plot(test_x, test_y, color='red', label='test acc') 
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    # x_major_locator=MultipleLocator(1)
    # #把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator=MultipleLocator(10)
    # #把y轴的刻度间隔设置为10，并存在变量里
    # ax=plt.gca()
    # #ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # #把x轴的主刻度设置为1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # #把y轴的主刻度设置为10的倍数    
    
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()

def SaveLossCurve(model,dataset,exp_result_dir,global_train_loss,global_test_loss,png_name):
    train_x = list(range(len(global_train_loss)))
    train_y = global_train_loss
    test_x = list(range(len(global_test_loss)))
    test_y = global_test_loss
    plt.title(f'{png_name}')
    plt.plot(train_x, train_y, color='black', label='train loss')     
    plt.plot(test_x, test_y, color='red', label='test loss') 

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()
    

def Save3AccuracyCurve(model,dataset,exp_result_dir,global_train_acc, global_cle_test_acc, global_adv_test_acc,png_name):
    train_x = list(range(len(global_train_acc)))
    train_y = global_train_acc
    cle_test_x = list(range(len(global_cle_test_acc)))
    cle_test_y = global_cle_test_acc
    adv_test_x = list(range(len(global_adv_test_acc)))
    adv_test_y = global_adv_test_acc    
    
    plt.title(f'{png_name}')
    plt.plot(train_x, train_y, color='black', label='train acc') 
    plt.plot(cle_test_x, cle_test_y, color='red', label='clean test acc') 
    plt.plot(adv_test_x, adv_test_y, color='blue', label='robust test acc') 
    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    # x_major_locator=MultipleLocator(1)
    # #把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator=MultipleLocator(10)
    # #把y轴的刻度间隔设置为10，并存在变量里
    # ax=plt.gca()
    # #ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # #把x轴的主刻度设置为1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # #把y轴的主刻度设置为10的倍数   
        
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()

def Save3LossCurve(model,dataset,exp_result_dir,global_train_loss, global_cle_test_loss, global_adv_test_loss,png_name):
    train_x = list(range(len(global_train_loss)))
    train_y = global_train_loss
    cle_test_x = list(range(len(global_cle_test_loss)))
    cle_test_y = global_cle_test_loss
    adv_test_x = list(range(len(global_adv_test_loss)))
    adv_test_y = global_adv_test_loss    
    
    plt.title(f'{png_name}')
    plt.plot(train_x, train_y, color='black', label='train loss')                 #   TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    # plt.plot(train_x.cpu(), train_y, color='black', label='on trainset loss')         #   AttributeError: 'list' object has no attribute 'cpu'
    plt.plot(cle_test_x, cle_test_y, color='red', label='clean test loss') 
    # plt.plot(test_x.cpu(), test_y, color='red', label='on testset loss') 

    # plt.plot(train_x.cpu(), train_y.cpu(), color='black', label='on trainset loss') 
    # plt.plot(test_x.cpu(), test_y.cpu(), color='red', label='on testset loss')     
    plt.plot(adv_test_x, adv_test_y, color='blue', label='robust test loss') 
    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()