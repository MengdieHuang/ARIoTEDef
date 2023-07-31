import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FN_advmal_9083_file = pd.read_csv('9083-on-FN-cle-mal.csv')
FN_advmal_9083_df = pd.DataFrame(FN_advmal_9083_file)
FN_advmal_9083_df = FN_advmal_9083_df.drop(columns='Wall time')  
FN_advmal_9083_df['Size']=9083    

FN_advmal_5000_file = pd.read_csv('5000-on-FN-cle-mal.csv')
FN_advmal_5000_df = pd.DataFrame(FN_advmal_5000_file)
FN_advmal_5000_df = FN_advmal_5000_df.drop(columns='Wall time')                   
FN_advmal_5000_df['Size']=5000    

FN_advmal_2500_file = pd.read_csv('2500-on-FN-cle-mal.csv')
FN_advmal_2500_df = pd.DataFrame(FN_advmal_2500_file)
FN_advmal_2500_df = FN_advmal_2500_df.drop(columns='Wall time')                   
FN_advmal_2500_df['Size']=2500    

FN_advmal_1000_file = pd.read_csv('1000-on-FN-cle-mal.csv')
FN_advmal_1000_df = pd.DataFrame(FN_advmal_1000_file)      
FN_advmal_1000_df = FN_advmal_1000_df.drop(columns='Wall time')                   
FN_advmal_1000_df['Size']=1000    

FN_advmal_500_file = pd.read_csv('500-on-FN-cle-mal.csv')
FN_advmal_500_df = pd.DataFrame(FN_advmal_500_file)
FN_advmal_500_df = FN_advmal_500_df.drop(columns='Wall time')                   
FN_advmal_500_df['Size']=500    

FN_advmal_250_file = pd.read_csv('250-on-FN-cle-mal.csv')
FN_advmal_250_df = pd.DataFrame(FN_advmal_250_file)
FN_advmal_250_df = FN_advmal_250_df.drop(columns='Wall time')                   
FN_advmal_250_df['Size']=250    

FN_advmal_df = pd.concat([FN_advmal_9083_df, FN_advmal_5000_df, FN_advmal_2500_df, FN_advmal_1000_df,FN_advmal_500_df, FN_advmal_250_df])
FN_advmal_df = FN_advmal_df.reset_index(drop=True)
print("FN_advmal_df:\n",FN_advmal_df)  

# sns.set_theme(style="darkgrid")
plt.style.use('seaborn')    
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True,dpi=500) 

sns_plot=sns.lineplot(data=FN_advmal_9083_df, x='Step', y='Value', label=f'RetrainSetSize=9083')
sns_plot=sns.lineplot(data=FN_advmal_5000_df, x='Step', y='Value', label=f'RetrainSetSize=5000')
sns_plot=sns.lineplot(data=FN_advmal_2500_df, x='Step', y='Value', label=f'RetrainSetSize=2500')
sns_plot=sns.lineplot(data=FN_advmal_1000_df, x='Step', y='Value', label=f'RetrainSetSize=1000')
sns_plot=sns.lineplot(data=FN_advmal_500_df, x='Step', y='Value', label=f'RetrainSetSize=500')
sns_plot=sns.lineplot(data=FN_advmal_250_df, x='Step', y='Value', label=f'RetrainSetSize=250')

plt.xlabel('Round', fontsize=12)
plt.ylabel('False Negative Value', fontsize=12)
plt.title('FN on Clean Malicious Traffic', fontsize=14);  
plt.legend(
    [
    f'RetrainSetSize=9083', 
    f'RetrainSetSize=5000',
    f'RetrainSetSize=2500',
    f'RetrainSetSize=1000', 
    f'RetrainSetSize=500',
    f'RetrainSetSize=250',    
    ], 
    fontsize=8, loc='best') 

savepath = f'/home/huan1932/NID/cisco-nid/figures/new/advnum-effect/effect-on-FN-cle-mal'
savename = f'retrain-advnum-effect-on-FN-on-cle-mal-seaborn'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

            
print("shadow--------------------")            

plt.style.use('seaborn')    
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率
sns.lineplot(data=FN_advmal_df, x='Step', y='Value')      
plt.xlabel('Round', fontsize=12)
plt.ylabel('False Negative Value', fontsize=12)
plt.title('FN on Clean Malicious Traffic', fontsize=14);  
      
savepath = f'/home/huan1932/NID/cisco-nid/figures/new/advnum-effect/effect-on-FN-cle-mal'
savename = f'retrain-advnum-effect-on-FN-on-cle-mal-seaborn-shadow'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close 