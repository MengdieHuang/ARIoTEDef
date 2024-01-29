import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

advset2_rounds051020_on_advmal_file = pd.read_csv('advset-2-white-robustness-average.csv')
advset2_rounds051020_on_advmal_df = pd.DataFrame(advset2_rounds051020_on_advmal_file)
print("advset2_rounds051020_on_advmal_df:\n",advset2_rounds051020_on_advmal_df)  

advset2_rounds051020_TP_on_advmal_df = advset2_rounds051020_on_advmal_file.drop(columns=['FN','FNR', 'Recall'])  
# advset2_rounds051020_TP_on_advmal_df.rename(columns={'TP': 'Value'}, inplace=True)
# advset2_rounds051020_TP_on_advmal_df['TP_FN_label']='TP'    
print("advset2_rounds051020_TP_on_advmal_df:\n",advset2_rounds051020_TP_on_advmal_df)  

advset2_rounds051020_FN_on_advmal_df = advset2_rounds051020_on_advmal_file.drop(columns=['TP','FNR', 'Recall'])  
# advset2_rounds051020_FN_on_advmal_df.rename(columns={'FN': 'Value'}, inplace=True)
# advset2_rounds051020_FN_on_advmal_df['TP_FN_label']='FN'   
print("advset2_rounds051020_FN_on_advmal_df:\n",advset2_rounds051020_FN_on_advmal_df)  

# advset2_rounds051020_TPFN_on_advma_merge_df = pd.concat([advset2_rounds051020_FN_on_advmal_df,advset2_rounds051020_TP_on_advmal_df])
# print("advset2_rounds051020_TPFN_on_advma_merge_df:\n",advset2_rounds051020_TPFN_on_advma_merge_df)  

# advset2_rounds051020_FNR_on_advmal_df = advset2_rounds051020_on_advmal_file.drop(columns=['TP','FN', 'Recall'])  
# advset2_rounds051020_Recall_on_advmal_df = advset2_rounds051020_on_advmal_file.drop(columns=['TP','FN', 'FNR'])  

plt.style.use('seaborn') 

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

# sns_plot=sns.barplot(data=advset2_rounds051020_TPFN_on_advma_merge_df, x='size', y='Value', hue="TP_FN_label", palette="Set1")
sns_plot=sns.barplot(data=advset2_rounds051020_FN_on_advmal_df, x='round', y='FN', hue="model", palette="Set1")

ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_ylabel('FN Value', fontsize=12)
plt.legend(title=None)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax.set_title('Adversarial Setting II', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset2/compare-rounds'
savename = f'bar-figure-advset2-round051020-average-FN-performance'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#--------------------------------------
plt.style.use('seaborn') 

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

# sns_plot=sns.barplot(data=advset2_rounds051020_TPFN_on_advma_merge_df, x='size', y='Value', hue="TP_FN_label", palette="Set1")
sns_plot=sns.barplot(data=advset2_rounds051020_TP_on_advmal_df, x='round', y='TP', hue="model", palette="Set1")

ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_ylabel('TP Value', fontsize=12)
plt.legend(title=None)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax.set_title('Adversarial Setting II', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset2/compare-rounds'
savename = f'bar-figure-advset2-round051020-average-TP-performance'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   