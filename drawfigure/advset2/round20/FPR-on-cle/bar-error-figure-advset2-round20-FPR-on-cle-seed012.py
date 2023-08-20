import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

advset2_rounds20_preevolved_FPR_on_cle_file = pd.read_csv('test_FPrate_on_former_set_list_advset2_round20_seed012.csv')
advset2_rounds20_preevolved_FPR_on_cle_df = pd.DataFrame(advset2_rounds20_preevolved_FPR_on_cle_file)

advset2_rounds20_preevolved_FPR_on_cle_seed_0_df = advset2_rounds20_preevolved_FPR_on_cle_df.drop(columns=['seed1','seed2'])  
advset2_rounds20_preevolved_FPR_on_cle_seed_0_df.rename(columns={'seed0': 'FPR'}, inplace=True)

advset2_rounds20_preevolved_FPR_on_cle_seed_1_df = advset2_rounds20_preevolved_FPR_on_cle_df.drop(columns=['seed0','seed2'])  
advset2_rounds20_preevolved_FPR_on_cle_seed_1_df.rename(columns={'seed1': 'FPR'}, inplace=True)

advset2_rounds20_preevolved_FPR_on_cle_seed_2_df = advset2_rounds20_preevolved_FPR_on_cle_df.drop(columns=['seed0','seed1'])  
advset2_rounds20_preevolved_FPR_on_cle_seed_2_df.rename(columns={'seed2': 'FPR'}, inplace=True)

print("advset2_rounds20_preevolved_FPR_on_cle_df:\n",advset2_rounds20_preevolved_FPR_on_cle_df)  
print("advset2_rounds20_preevolved_FPR_on_cle_seed_0_df:\n",advset2_rounds20_preevolved_FPR_on_cle_seed_0_df)  
print("advset2_rounds20_preevolved_FPR_on_cle_seed_1_df:\n",advset2_rounds20_preevolved_FPR_on_cle_seed_1_df)  
print("advset2_rounds20_preevolved_FPR_on_cle_seed_2_df:\n",advset2_rounds20_preevolved_FPR_on_cle_seed_2_df)  



advset2_rounds20_preevolved_FPR_on_cle_merge_df = pd.concat([advset2_rounds20_preevolved_FPR_on_cle_seed_0_df, advset2_rounds20_preevolved_FPR_on_cle_seed_1_df, advset2_rounds20_preevolved_FPR_on_cle_seed_2_df])
# advset2_rounds20_preevolved_FPR_on_cle_merge_df = pd.concat([advset2_rounds20_preevolved_FPR_on_cle_df, advset2_rounds20_preevolved_FPR_on_cle_df, advset2_rounds20_preevolved_FPR_on_cle_df])

print("advset2_rounds20_preevolved_FPR_on_cle_merge_df:\n",advset2_rounds20_preevolved_FPR_on_cle_merge_df)  

advset2_rounds20_preevolved_FPR_on_cle_merge_df = advset2_rounds20_preevolved_FPR_on_cle_merge_df.reset_index(drop=True)

print("advset2_rounds20_preevolved_FPR_on_cle_merge_df:\n",advset2_rounds20_preevolved_FPR_on_cle_merge_df)  

# advset2_rounds20_preevolved_FPR_on_cle_merge_df.rename()

# advset2_rounds201020_TP_on_advmal_df.rename(columns={'TP': 'Value'}, inplace=True)
advset2_rounds20_preevolved_FPR_on_cle_merge_df['evolution_label']='Before the current round'    




# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


# sns_plot=sns.lineplot(data=advset2_rounds20_preevolved_FPR_on_cle_merge_df, x='round', y='FPR', color='b')
sns_plot=sns.barplot(data=advset2_rounds20_preevolved_FPR_on_cle_merge_df, x='round', y='FPR', color='b')

ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_ylabel('FPR (%) on Clean Test Set', fontsize=12)
ax.set_title('Adversarial Setting II', fontsize=14); 
plt.ylim(-5, 105)
# plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset2/round20/FPR-on-cle'
# savename = f'line-shadow-figure-advset2-round20-FPR-on-adv-seed012'
savename = f'bar-error-figure-advset2-round20-preevolved_FPR-on-cle-seed012'

plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   














#==========================================================
advset2_rounds20_evolved_FPR_on_cle_file = pd.read_csv('test_FPrate_on_later_set_list_advset2_round20_seed012.csv')
advset2_rounds20_evolved_FPR_on_cle_df = pd.DataFrame(advset2_rounds20_evolved_FPR_on_cle_file)


# 删除最后一行
advset2_rounds20_evolved_FPR_on_cle_df = advset2_rounds20_evolved_FPR_on_cle_df.drop(advset2_rounds20_evolved_FPR_on_cle_df.index[-1])
# x从round1开始显示
advset2_rounds20_evolved_FPR_on_cle_df['round'] = advset2_rounds20_evolved_FPR_on_cle_df['round'] + 1



advset2_rounds20_evolved_FPR_on_cle_seed_0_df = advset2_rounds20_evolved_FPR_on_cle_df.drop(columns=['seed1','seed2'])  
advset2_rounds20_evolved_FPR_on_cle_seed_0_df.rename(columns={'seed0': 'FPR'}, inplace=True)

advset2_rounds20_evolved_FPR_on_cle_seed_1_df = advset2_rounds20_evolved_FPR_on_cle_df.drop(columns=['seed0','seed2'])  
advset2_rounds20_evolved_FPR_on_cle_seed_1_df.rename(columns={'seed1': 'FPR'}, inplace=True)

advset2_rounds20_evolved_FPR_on_cle_seed_2_df = advset2_rounds20_evolved_FPR_on_cle_df.drop(columns=['seed0','seed1'])  
advset2_rounds20_evolved_FPR_on_cle_seed_2_df.rename(columns={'seed2': 'FPR'}, inplace=True)

print("advset2_rounds20_evolved_FPR_on_cle_df:\n",advset2_rounds20_evolved_FPR_on_cle_df)  
print("advset2_rounds20_evolved_FPR_on_cle_seed_0_df:\n",advset2_rounds20_evolved_FPR_on_cle_seed_0_df)  
print("advset2_rounds20_evolved_FPR_on_cle_seed_1_df:\n",advset2_rounds20_evolved_FPR_on_cle_seed_1_df)  
print("advset2_rounds20_evolved_FPR_on_cle_seed_2_df:\n",advset2_rounds20_evolved_FPR_on_cle_seed_2_df)  



advset2_rounds20_evolved_FPR_on_cle_merge_df = pd.concat([advset2_rounds20_evolved_FPR_on_cle_seed_0_df, advset2_rounds20_evolved_FPR_on_cle_seed_1_df, advset2_rounds20_evolved_FPR_on_cle_seed_2_df])
# advset2_rounds20_evolved_FPR_on_cle_merge_df = pd.concat([advset2_rounds20_evolved_FPR_on_cle_df, advset2_rounds20_evolved_FPR_on_cle_df, advset2_rounds20_evolved_FPR_on_cle_df])

print("advset2_rounds20_evolved_FPR_on_cle_merge_df:\n",advset2_rounds20_evolved_FPR_on_cle_merge_df)  

advset2_rounds20_evolved_FPR_on_cle_merge_df = advset2_rounds20_evolved_FPR_on_cle_merge_df.reset_index(drop=True)

print("advset2_rounds20_evolved_FPR_on_cle_merge_df:\n",advset2_rounds20_evolved_FPR_on_cle_merge_df)  

advset2_rounds20_evolved_FPR_on_cle_merge_df['evolution_label']='After the current round'    




# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


# sns_plot=sns.lineplot(data=advset2_rounds20_evolved_FPR_on_cle_merge_df, x='round', y='FPR', color='b')

sns_plot=sns.barplot(data=advset2_rounds20_evolved_FPR_on_cle_merge_df, x='round', y='FPR', color='b')

ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_ylabel('FPR (%) on Clean Test Set', fontsize=12)
ax.set_title('Adversarial Setting II', fontsize=14); 
plt.ylim(-5, 105)
# plt.xlim(1, 105)

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset2/round20/FPR-on-cle'
# savename = f'line-shadow-figure-advset2-round20-FPR-on-adv-seed012'
savename = f'bar-error-figure-advset2-round20-evolved_FPR-on-cle-seed012'

plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   


#======================================================


















advset2_rounds20_FPR_on_cle_merge_df = pd.concat([advset2_rounds20_preevolved_FPR_on_cle_merge_df,advset2_rounds20_evolved_FPR_on_cle_merge_df])





plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.5), tight_layout=True, dpi=500)

sns_plot=sns.barplot(data=advset2_rounds20_FPR_on_cle_merge_df, x='round', y='FPR', hue="evolution_label", palette="husl",errorbar="sd")
# sns_plot=sns.boxplot(data=advset2_rounds20_FPR_on_cle_merge_df, x='round', y='FPR', hue="evolution_label", palette="tab10")

ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_ylabel('FPR (%) on Clean Test Set', fontsize=12)
ax.set_title('Adversarial Setting II', fontsize=14); 
plt.ylim(0, 105)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='upper left',frameon=True)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))


savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset2/round20/FPR-on-cle'
savename = f'bar-error-figure-advset2-round20-FPR-on-cle-seed012'

plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   