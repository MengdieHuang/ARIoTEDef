import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

infection_retrain_epo40_costtime_file = pd.read_csv('infection_detector_retrain_time_list_round20_epoch40.csv')
infection_retrain_epo40_costtime_df = pd.DataFrame(infection_retrain_epo40_costtime_file)

infection_retrain_epo40_costtime_seed_0_df = infection_retrain_epo40_costtime_df.drop(columns=['seed1','seed2'])  
infection_retrain_epo40_costtime_seed_0_df.rename(columns={'seed0': 'costtime'}, inplace=True)

infection_retrain_epo40_costtime_seed_1_df = infection_retrain_epo40_costtime_df.drop(columns=['seed0','seed2'])  
infection_retrain_epo40_costtime_seed_1_df.rename(columns={'seed1': 'costtime'}, inplace=True)

infection_retrain_epo40_costtime_seed_2_df = infection_retrain_epo40_costtime_df.drop(columns=['seed0','seed1'])  
infection_retrain_epo40_costtime_seed_2_df.rename(columns={'seed2': 'costtime'}, inplace=True)

print("infection_retrain_epo40_costtime_df:\n",infection_retrain_epo40_costtime_df)  
print("infection_retrain_epo40_costtime_seed_0_df:\n",infection_retrain_epo40_costtime_seed_0_df)  
print("infection_retrain_epo40_costtime_seed_1_df:\n",infection_retrain_epo40_costtime_seed_1_df)  
print("infection_retrain_epo40_costtime_seed_2_df:\n",infection_retrain_epo40_costtime_seed_2_df)  



infection_retrain_epo40_costtime_merge_df = pd.concat([infection_retrain_epo40_costtime_seed_0_df, infection_retrain_epo40_costtime_seed_1_df, infection_retrain_epo40_costtime_seed_2_df])
# infection_retrain_epo40_costtime_merge_df = pd.concat([infection_retrain_epo40_costtime_df, infection_retrain_epo40_costtime_df, infection_retrain_epo40_costtime_df])

print("infection_retrain_epo40_costtime_merge_df:\n",infection_retrain_epo40_costtime_merge_df)  

infection_retrain_epo40_costtime_merge_df = infection_retrain_epo40_costtime_merge_df.reset_index(drop=True)

print("infection_retrain_epo40_costtime_merge_df:\n",infection_retrain_epo40_costtime_merge_df)  
#================================================




infection_retrain_epo30_costtime_file = pd.read_csv('infection_detector_retrain_time_list_round20_epoch30.csv')
infection_retrain_epo30_costtime_df = pd.DataFrame(infection_retrain_epo30_costtime_file)

infection_retrain_epo30_costtime_seed_0_df = infection_retrain_epo30_costtime_df.drop(columns=['seed1','seed2'])  
infection_retrain_epo30_costtime_seed_0_df.rename(columns={'seed0': 'costtime'}, inplace=True)

infection_retrain_epo30_costtime_seed_1_df = infection_retrain_epo30_costtime_df.drop(columns=['seed0','seed2'])  
infection_retrain_epo30_costtime_seed_1_df.rename(columns={'seed1': 'costtime'}, inplace=True)

infection_retrain_epo30_costtime_seed_2_df = infection_retrain_epo30_costtime_df.drop(columns=['seed0','seed1'])  
infection_retrain_epo30_costtime_seed_2_df.rename(columns={'seed2': 'costtime'}, inplace=True)

print("infection_retrain_epo30_costtime_df:\n",infection_retrain_epo30_costtime_df)  
print("infection_retrain_epo30_costtime_seed_0_df:\n",infection_retrain_epo30_costtime_seed_0_df)  
print("infection_retrain_epo30_costtime_seed_1_df:\n",infection_retrain_epo30_costtime_seed_1_df)  
print("infection_retrain_epo30_costtime_seed_2_df:\n",infection_retrain_epo30_costtime_seed_2_df)  



infection_retrain_epo30_costtime_merge_df = pd.concat([infection_retrain_epo30_costtime_seed_0_df, infection_retrain_epo30_costtime_seed_1_df, infection_retrain_epo30_costtime_seed_2_df])
# infection_retrain_epo30_costtime_merge_df = pd.concat([infection_retrain_epo30_costtime_df, infection_retrain_epo30_costtime_df, infection_retrain_epo30_costtime_df])

print("infection_retrain_epo30_costtime_merge_df:\n",infection_retrain_epo30_costtime_merge_df)  

infection_retrain_epo30_costtime_merge_df = infection_retrain_epo30_costtime_merge_df.reset_index(drop=True)

print("infection_retrain_epo30_costtime_merge_df:\n",infection_retrain_epo30_costtime_merge_df)  

#================================================




infection_retrain_epo20_costtime_file = pd.read_csv('infection_detector_retrain_time_list_round20_epoch20.csv')
infection_retrain_epo20_costtime_df = pd.DataFrame(infection_retrain_epo20_costtime_file)

infection_retrain_epo20_costtime_seed_0_df = infection_retrain_epo20_costtime_df.drop(columns=['seed1','seed2'])  
infection_retrain_epo20_costtime_seed_0_df.rename(columns={'seed0': 'costtime'}, inplace=True)

infection_retrain_epo20_costtime_seed_1_df = infection_retrain_epo20_costtime_df.drop(columns=['seed0','seed2'])  
infection_retrain_epo20_costtime_seed_1_df.rename(columns={'seed1': 'costtime'}, inplace=True)

infection_retrain_epo20_costtime_seed_2_df = infection_retrain_epo20_costtime_df.drop(columns=['seed0','seed1'])  
infection_retrain_epo20_costtime_seed_2_df.rename(columns={'seed2': 'costtime'}, inplace=True)

print("infection_retrain_epo20_costtime_df:\n",infection_retrain_epo20_costtime_df)  
print("infection_retrain_epo20_costtime_seed_0_df:\n",infection_retrain_epo20_costtime_seed_0_df)  
print("infection_retrain_epo20_costtime_seed_1_df:\n",infection_retrain_epo20_costtime_seed_1_df)  
print("infection_retrain_epo20_costtime_seed_2_df:\n",infection_retrain_epo20_costtime_seed_2_df)  



infection_retrain_epo20_costtime_merge_df = pd.concat([infection_retrain_epo20_costtime_seed_0_df, infection_retrain_epo20_costtime_seed_1_df, infection_retrain_epo20_costtime_seed_2_df])
# infection_retrain_epo20_costtime_merge_df = pd.concat([infection_retrain_epo20_costtime_df, infection_retrain_epo20_costtime_df, infection_retrain_epo20_costtime_df])

print("infection_retrain_epo20_costtime_merge_df:\n",infection_retrain_epo20_costtime_merge_df)  

infection_retrain_epo20_costtime_merge_df = infection_retrain_epo20_costtime_merge_df.reset_index(drop=True)

print("infection_retrain_epo20_costtime_merge_df:\n",infection_retrain_epo20_costtime_merge_df)  

#================================================

# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


sns_plot=sns.lineplot(data=infection_retrain_epo40_costtime_merge_df, x='round', y='costtime', color='b')



sns_plot=sns.lineplot(data=infection_retrain_epo40_costtime_merge_df, x='round', y='costtime', color='b', label='40 epochs per round')

sns_plot=sns.lineplot(data=infection_retrain_epo30_costtime_merge_df, x='round', y='costtime', color='r', label='30 epochs per round')

sns_plot=sns.lineplot(data=infection_retrain_epo20_costtime_merge_df, x='round', y='costtime', color='g', label='20 epochs per round')





ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_ylabel('Cost Time (seconds)', fontsize=12)
ax.set_title('Infection Detector Evolve', fontsize=14); 
# plt.ylim(0.5, 2.0)
plt.ylim(0, 80)
# 调整横坐标显示范围
plt.xlim(0, 21)
# plt.legend(loc='best')
plt.legend(loc='lower right')
# ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))


savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.6/test-time/infecDec-evolve/different-epoch'
savename = f'line-shadow-figure-infection-retrain-costtime-seed012-epo203040'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   
