import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

advset1_rounds204080_on_advmal_file = pd.read_csv('average_adv_test_TP_FN_FNR_Recall_list_advset1_round204080.csv')
advset1_rounds204080_on_advmal_df = pd.DataFrame(advset1_rounds204080_on_advmal_file)

advset1_rounds204080_TP_on_advmal_df = advset1_rounds204080_on_advmal_file.drop(columns=['FN','FNR', 'Recall'])  
advset1_rounds204080_TP_on_advmal_df.rename(columns={'TP': 'Value'}, inplace=True)
advset1_rounds204080_TP_on_advmal_df['TP_FN_label']='TP'    

advset1_rounds204080_FN_on_advmal_df = advset1_rounds204080_on_advmal_file.drop(columns=['TP','FNR', 'Recall'])  
advset1_rounds204080_FN_on_advmal_df.rename(columns={'FN': 'Value'}, inplace=True)
advset1_rounds204080_FN_on_advmal_df['TP_FN_label']='FN'   

print("advset1_rounds204080_on_advmal_df:\n",advset1_rounds204080_on_advmal_df)  
print("advset1_rounds204080_TP_on_advmal_df:\n",advset1_rounds204080_TP_on_advmal_df)  
print("advset1_rounds204080_FN_on_advmal_df:\n",advset1_rounds204080_FN_on_advmal_df)  

advset1_rounds204080_TPFN_on_advma_merge_df = pd.concat([advset1_rounds204080_FN_on_advmal_df,advset1_rounds204080_TP_on_advmal_df])
# advset1_rounds204080_TPFN_on_advma_merge_df = advset1_rounds204080_TPFN_on_advma_merge_df.reset_index(drop=True)
print("advset1_rounds204080_TPFN_on_advma_merge_df:\n",advset1_rounds204080_TPFN_on_advma_merge_df)  


advset1_rounds204080_FNR_on_advmal_df = advset1_rounds204080_on_advmal_file.drop(columns=['TP','FN', 'Recall'])  
# advset1_rounds204080_FNR_on_advmal_df.rename(columns={'seed0': 'costtime'}, inplace=True)
advset1_rounds204080_Recall_on_advmal_df = advset1_rounds204080_on_advmal_file.drop(columns=['TP','FN', 'FNR'])  
# advset1_rounds204080_Recall_on_advmal_df.rename(columns={'seed0': 'costtime'}, inplace=True)
# print("advset1_rounds204080_FNR_on_advmal_df:\n",advset1_rounds204080_FNR_on_advmal_df)  



# advset1_rounds80_costtime_on_advmal_merge_df = pd.concat([advset1_rounds80_costtime_on_advmal_seed_0_df, advset1_rounds80_costtime_on_advmal_seed_1_df, advset1_rounds80_costtime_on_advmal_seed_2_df])
# advset1_rounds80_costtime_on_advmal_merge_df = pd.concat([advset1_rounds80_costtime_on_advmal_df, advset1_rounds80_costtime_on_advmal_df, advset1_rounds80_costtime_on_advmal_df])

# print("advset1_rounds80_costtime_on_advmal_merge_df:\n",advset1_rounds80_costtime_on_advmal_merge_df)  

# advset1_rounds80_costtime_on_advmal_merge_df = advset1_rounds80_costtime_on_advmal_merge_df.reset_index(drop=True)

# print("advset1_rounds80_costtime_on_advmal_merge_df:\n",advset1_rounds80_costtime_on_advmal_merge_df)  


# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
# _, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

# sns_plot=sns.barplot(data=advset1_rounds204080_TP_on_advmal_df, x='round', y='TP', label='TP', color='g')
# sns_plot=sns.barplot(data=advset1_rounds204080_FN_on_advmal_df, x='round', y='FN', label='FN', color='o')
# 自定义调色板颜色
# custom_palette = {'TP': 'blue', 'FN': 'red'}
# sns_plot=sns.barplot(data=advset1_rounds204080_TPFN_on_advma_merge_df, x='round', y='Value', hue="TP_FN_label", palette=custom_palette)
sns_plot=sns.barplot(data=advset1_rounds204080_TPFN_on_advma_merge_df, x='round', y='Value', hue="TP_FN_label", palette="Set1")

# sns_plot=sns.barplot(data=advset1_rounds204080_on_advmal_df, x='round', y='FN', label='FN', color='o')

# sns.barplot(data=df, x="island", y="body_mass_g", hue="TP_FN_label")


ax.set_xlabel('Evolved Model', fontsize=12)
ax.set_ylabel('FN Value / TP Value', fontsize=12)
# plt.legend(
#     fontsize=8, 
#     loc='upper right') 
plt.legend(title=None)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax.set_title('Adversarial Setting I', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset1/compare-rounds'
savename = f'bar-figure-advset1-round204080-average-TP-FN-performance'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   
