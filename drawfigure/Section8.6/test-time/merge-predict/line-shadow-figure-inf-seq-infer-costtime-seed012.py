import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

infection_infer_costtime_file = pd.read_csv('infection_detector_infer_time_list.csv')
infection_infer_costtime_df = pd.DataFrame(infection_infer_costtime_file)

infection_infer_costtime_seed_0_df = infection_infer_costtime_df.drop(columns=['seed1','seed2'])  
infection_infer_costtime_seed_0_df.rename(columns={'seed0': 'costtime'}, inplace=True)

infection_infer_costtime_seed_1_df = infection_infer_costtime_df.drop(columns=['seed0','seed2'])  
infection_infer_costtime_seed_1_df.rename(columns={'seed1': 'costtime'}, inplace=True)

infection_infer_costtime_seed_2_df = infection_infer_costtime_df.drop(columns=['seed0','seed1'])  
infection_infer_costtime_seed_2_df.rename(columns={'seed2': 'costtime'}, inplace=True)

print("infection_infer_costtime_df:\n",infection_infer_costtime_df)  
print("infection_infer_costtime_seed_0_df:\n",infection_infer_costtime_seed_0_df)  
print("infection_infer_costtime_seed_1_df:\n",infection_infer_costtime_seed_1_df)  
print("infection_infer_costtime_seed_2_df:\n",infection_infer_costtime_seed_2_df)  



infection_infer_costtime_merge_df = pd.concat([infection_infer_costtime_seed_0_df, infection_infer_costtime_seed_1_df, infection_infer_costtime_seed_2_df])
# infection_infer_costtime_merge_df = pd.concat([infection_infer_costtime_df, infection_infer_costtime_df, infection_infer_costtime_df])

print("infection_infer_costtime_merge_df:\n",infection_infer_costtime_merge_df)  

infection_infer_costtime_merge_df = infection_infer_costtime_merge_df.reset_index(drop=True)

print("infection_infer_costtime_merge_df:\n",infection_infer_costtime_merge_df)  
#=======================================================

s2sanalyzer_infer_costtime_file = pd.read_csv('s2sanalyzer_infer_time_list.csv')
s2sanalyzer_infer_costtime_df = pd.DataFrame(s2sanalyzer_infer_costtime_file)

s2sanalyzer_infer_costtime_seed_0_df = s2sanalyzer_infer_costtime_df.drop(columns=['seed1','seed2'])  
s2sanalyzer_infer_costtime_seed_0_df.rename(columns={'seed0': 'costtime'}, inplace=True)

s2sanalyzer_infer_costtime_seed_1_df = s2sanalyzer_infer_costtime_df.drop(columns=['seed0','seed2'])  
s2sanalyzer_infer_costtime_seed_1_df.rename(columns={'seed1': 'costtime'}, inplace=True)

s2sanalyzer_infer_costtime_seed_2_df = s2sanalyzer_infer_costtime_df.drop(columns=['seed0','seed1'])  
s2sanalyzer_infer_costtime_seed_2_df.rename(columns={'seed2': 'costtime'}, inplace=True)

print("s2sanalyzer_infer_costtime_df:\n",s2sanalyzer_infer_costtime_df)  
print("s2sanalyzer_infer_costtime_seed_0_df:\n",s2sanalyzer_infer_costtime_seed_0_df)  
print("s2sanalyzer_infer_costtime_seed_1_df:\n",s2sanalyzer_infer_costtime_seed_1_df)  
print("s2sanalyzer_infer_costtime_seed_2_df:\n",s2sanalyzer_infer_costtime_seed_2_df)  



s2sanalyzer_infer_costtime_merge_df = pd.concat([s2sanalyzer_infer_costtime_seed_0_df, s2sanalyzer_infer_costtime_seed_1_df, s2sanalyzer_infer_costtime_seed_2_df])
# s2sanalyzer_infer_costtime_merge_df = pd.concat([s2sanalyzer_infer_costtime_df, s2sanalyzer_infer_costtime_df, s2sanalyzer_infer_costtime_df])

print("s2sanalyzer_infer_costtime_merge_df:\n",s2sanalyzer_infer_costtime_merge_df)  

s2sanalyzer_infer_costtime_merge_df = s2sanalyzer_infer_costtime_merge_df.reset_index(drop=True)

print("s2sanalyzer_infer_costtime_merge_df:\n",s2sanalyzer_infer_costtime_merge_df)  
#=======================================================



# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


# sns_plot=sns.lineplot(data=infection_infer_costtime_merge_df, x='round', y='costtime', color='b')


sns_plot=sns.lineplot(data=infection_infer_costtime_merge_df, x='round', y='costtime', color='b',label='Infection Detector')

sns_plot=sns.lineplot(data=s2sanalyzer_infer_costtime_merge_df, x='round', y='costtime', color='r', label='Sequence Analyzer')



ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_ylabel('Cost Time (seconds)', fontsize=12)
ax.set_title('Predict', fontsize=14); 
plt.ylim(0, 0.1)
# plt.ylim(0.005, 0.011)
# 调整横坐标显示范围
plt.xlim(0, 21)
# 调整横坐标显示范围
# plt.xlim(1, 40)
# 设置y轴刻度为科学记数法
# ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend(loc='best')

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.6/test-time/merge-predict'
savename = f'line-shadow-figure-inf-seq-infer-costtime-seed012'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   
