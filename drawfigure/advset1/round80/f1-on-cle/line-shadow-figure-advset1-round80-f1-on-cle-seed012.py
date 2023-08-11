import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

advset1_rounds80_F1_on_cle_file = pd.read_csv('test_F1_list_advset1_round80_seed012.csv')
advset1_rounds80_F1_on_cle_df = pd.DataFrame(advset1_rounds80_F1_on_cle_file)

advset1_rounds80_F1_on_cle_seed_0_df = advset1_rounds80_F1_on_cle_df.drop(columns=['seed1','seed2'])  
advset1_rounds80_F1_on_cle_seed_0_df.rename(columns={'seed0': 'F1'}, inplace=True)

advset1_rounds80_F1_on_cle_seed_1_df = advset1_rounds80_F1_on_cle_df.drop(columns=['seed0','seed2'])  
advset1_rounds80_F1_on_cle_seed_1_df.rename(columns={'seed1': 'F1'}, inplace=True)

advset1_rounds80_F1_on_cle_seed_2_df = advset1_rounds80_F1_on_cle_df.drop(columns=['seed0','seed1'])  
advset1_rounds80_F1_on_cle_seed_2_df.rename(columns={'seed2': 'F1'}, inplace=True)

print("advset1_rounds80_F1_on_cle_df:\n",advset1_rounds80_F1_on_cle_df)  
print("advset1_rounds80_F1_on_cle_seed_0_df:\n",advset1_rounds80_F1_on_cle_seed_0_df)  
print("advset1_rounds80_F1_on_cle_seed_1_df:\n",advset1_rounds80_F1_on_cle_seed_1_df)  
print("advset1_rounds80_F1_on_cle_seed_2_df:\n",advset1_rounds80_F1_on_cle_seed_2_df)  



advset1_rounds80_F1_on_cle_merge_df = pd.concat([advset1_rounds80_F1_on_cle_seed_0_df, advset1_rounds80_F1_on_cle_seed_1_df, advset1_rounds80_F1_on_cle_seed_2_df])
# advset1_rounds80_F1_on_cle_merge_df = pd.concat([advset1_rounds80_F1_on_cle_df, advset1_rounds80_F1_on_cle_df, advset1_rounds80_F1_on_cle_df])

print("advset1_rounds80_F1_on_cle_merge_df:\n",advset1_rounds80_F1_on_cle_merge_df)  

advset1_rounds80_F1_on_cle_merge_df = advset1_rounds80_F1_on_cle_merge_df.reset_index(drop=True)

print("advset1_rounds80_F1_on_cle_merge_df:\n",advset1_rounds80_F1_on_cle_merge_df)  


# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


sns_plot=sns.lineplot(data=advset1_rounds80_F1_on_cle_merge_df, x='round', y='F1', color='b')
ax.set_xlabel('Evolve Round', fontsize=12)
ax.set_ylabel('F1 (%) on Clean Test Set', fontsize=12)
plt.ylim(0, 100)

ax.set_title('Adversarial Setting I', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset1/round80/f1-on-cle'
savename = f'line-shadow-figure-advset1-round80-f1-on-cle-seed012'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   
