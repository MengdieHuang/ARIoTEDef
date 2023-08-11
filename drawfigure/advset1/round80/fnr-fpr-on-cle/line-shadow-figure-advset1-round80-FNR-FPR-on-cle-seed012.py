import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

advset1_rounds80_FNrate_on_cle_file = pd.read_csv('test_FNrate_list_advset1_round80_seed012.csv')

advset1_rounds80_FNrate_on_cle_df = pd.DataFrame(advset1_rounds80_FNrate_on_cle_file)

advset1_rounds80_FNrate_on_cle_seed_0_df = advset1_rounds80_FNrate_on_cle_df.drop(columns=['seed1','seed2'])  
advset1_rounds80_FNrate_on_cle_seed_0_df.rename(columns={'seed0': 'FNR'}, inplace=True)

advset1_rounds80_FNrate_on_cle_seed_1_df = advset1_rounds80_FNrate_on_cle_df.drop(columns=['seed0','seed2'])  
advset1_rounds80_FNrate_on_cle_seed_1_df.rename(columns={'seed1': 'FNR'}, inplace=True)

advset1_rounds80_FNrate_on_cle_seed_2_df = advset1_rounds80_FNrate_on_cle_df.drop(columns=['seed0','seed1'])  
advset1_rounds80_FNrate_on_cle_seed_2_df.rename(columns={'seed2': 'FNR'}, inplace=True)

print("advset1_rounds80_FNrate_on_cle_df:\n",advset1_rounds80_FNrate_on_cle_df)  
print("advset1_rounds80_FNrate_on_cle_seed_0_df:\n",advset1_rounds80_FNrate_on_cle_seed_0_df)  
print("advset1_rounds80_FNrate_on_cle_seed_1_df:\n",advset1_rounds80_FNrate_on_cle_seed_1_df)  
print("advset1_rounds80_FNrate_on_cle_seed_2_df:\n",advset1_rounds80_FNrate_on_cle_seed_2_df)  



advset1_rounds80_FNrate_on_cle_merge_df = pd.concat([advset1_rounds80_FNrate_on_cle_seed_0_df, advset1_rounds80_FNrate_on_cle_seed_1_df, advset1_rounds80_FNrate_on_cle_seed_2_df])
# advset1_rounds80_FNrate_on_cle_merge_df = pd.concat([advset1_rounds80_FNrate_on_cle_df, advset1_rounds80_FNrate_on_cle_df, advset1_rounds80_FNrate_on_cle_df])

print("advset1_rounds80_FNrate_on_cle_merge_df:\n",advset1_rounds80_FNrate_on_cle_merge_df)  

advset1_rounds80_FNrate_on_cle_merge_df = advset1_rounds80_FNrate_on_cle_merge_df.reset_index(drop=True)

print("advset1_rounds80_FNrate_on_cle_merge_df:\n",advset1_rounds80_FNrate_on_cle_merge_df)  


# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


sns_plot=sns.lineplot(data=advset1_rounds80_FNrate_on_cle_merge_df, x='round', y='FNR', color='b')
ax.set_xlabel('Evolve Round', fontsize=12)
ax.set_ylabel('FNR (%) on Clean Test Set', fontsize=12)
plt.ylim(0, 100)

ax.set_title('Adversarial Setting I', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset1/round80/fnr-fpr-on-cle'
savename = f'line-shadow-figure-advset1-round80-FNR-on-cle-seed012'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   


#========================================================================================================
advset1_rounds80_FPrate_on_cle_file = pd.read_csv('test_FPrate_list_advset1_round80_seed012.csv')

advset1_rounds80_FPrate_on_cle_df = pd.DataFrame(advset1_rounds80_FPrate_on_cle_file)

advset1_rounds80_FPrate_on_cle_seed_0_df = advset1_rounds80_FPrate_on_cle_df.drop(columns=['seed1','seed2'])  
advset1_rounds80_FPrate_on_cle_seed_0_df.rename(columns={'seed0': 'FPR'}, inplace=True)

advset1_rounds80_FPrate_on_cle_seed_1_df = advset1_rounds80_FPrate_on_cle_df.drop(columns=['seed0','seed2'])  
advset1_rounds80_FPrate_on_cle_seed_1_df.rename(columns={'seed1': 'FPR'}, inplace=True)

advset1_rounds80_FPrate_on_cle_seed_2_df = advset1_rounds80_FPrate_on_cle_df.drop(columns=['seed0','seed1'])  
advset1_rounds80_FPrate_on_cle_seed_2_df.rename(columns={'seed2': 'FPR'}, inplace=True)

print("advset1_rounds80_FPrate_on_cle_df:\n",advset1_rounds80_FPrate_on_cle_df)  
print("advset1_rounds80_FPrate_on_cle_seed_0_df:\n",advset1_rounds80_FPrate_on_cle_seed_0_df)  
print("advset1_rounds80_FPrate_on_cle_seed_1_df:\n",advset1_rounds80_FPrate_on_cle_seed_1_df)  
print("advset1_rounds80_FPrate_on_cle_seed_2_df:\n",advset1_rounds80_FPrate_on_cle_seed_2_df)  



advset1_rounds80_FPrate_on_cle_merge_df = pd.concat([advset1_rounds80_FPrate_on_cle_seed_0_df, advset1_rounds80_FPrate_on_cle_seed_1_df, advset1_rounds80_FPrate_on_cle_seed_2_df])
# advset1_rounds80_FPrate_on_cle_merge_df = pd.concat([advset1_rounds80_FPrate_on_cle_df, advset1_rounds80_FPrate_on_cle_df, advset1_rounds80_FPrate_on_cle_df])

print("advset1_rounds80_FPrate_on_cle_merge_df:\n",advset1_rounds80_FPrate_on_cle_merge_df)  

advset1_rounds80_FPrate_on_cle_merge_df = advset1_rounds80_FPrate_on_cle_merge_df.reset_index(drop=True)

print("advset1_rounds80_FPrate_on_cle_merge_df:\n",advset1_rounds80_FPrate_on_cle_merge_df)  


# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


sns_plot=sns.lineplot(data=advset1_rounds80_FPrate_on_cle_merge_df, x='round', y='FPR', color='g')
ax.set_xlabel('Evolve Round', fontsize=12)
ax.set_ylabel('FPR (%) on Clean Test Set', fontsize=12)
plt.ylim(0, 100)

ax.set_title('Adversarial Setting I', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset1/round80/fnr-fpr-on-cle'
savename = f'line-shadow-figure-advset1-round80-FPR-on-cle-seed012'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#========================================================================================================

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


sns_plot=sns.lineplot(data=advset1_rounds80_FNrate_on_cle_merge_df, x='round', y='FNR', label='FNR', color='b')
sns_plot=sns.lineplot(data=advset1_rounds80_FPrate_on_cle_merge_df, x='round', y='FPR', label='FPR',color='g')

plt.legend(
    [f'FNR',
     f'FPR'
     ],
    fontsize=8, 
    loc='upper right') 
plt.legend(title=None)


ax.set_xlabel('Evolve Round', fontsize=12)
ax.set_ylabel('FNR and FPR (%) on Clean Test Set', fontsize=12)
plt.ylim(0, 100)

ax.set_title('Adversarial Setting I', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset1/round80/fnr-fpr-on-cle'
savename = f'line-shadow-figure-advset1-round80-FNR-FPR-on-cle-seed012'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   