import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

stdtrain_seq2seq_40epoch_costtime_file = pd.read_csv('stdtrain-seq2seq-40epoch-costtime-seed012.csv')
stdtrain_seq2seq_40epoch_costtime_df = pd.DataFrame(stdtrain_seq2seq_40epoch_costtime_file)

stdtrain_seq2seq_40epoch_costtime_seed_0_df = stdtrain_seq2seq_40epoch_costtime_df.drop(columns=['seed1','seed2'])  
stdtrain_seq2seq_40epoch_costtime_seed_0_df.rename(columns={'seed0': 'costtime'}, inplace=True)

stdtrain_seq2seq_40epoch_costtime_seed_1_df = stdtrain_seq2seq_40epoch_costtime_df.drop(columns=['seed0','seed2'])  
stdtrain_seq2seq_40epoch_costtime_seed_1_df.rename(columns={'seed1': 'costtime'}, inplace=True)

stdtrain_seq2seq_40epoch_costtime_seed_2_df = stdtrain_seq2seq_40epoch_costtime_df.drop(columns=['seed0','seed1'])  
stdtrain_seq2seq_40epoch_costtime_seed_2_df.rename(columns={'seed2': 'costtime'}, inplace=True)

print("stdtrain_seq2seq_40epoch_costtime_df:\n",stdtrain_seq2seq_40epoch_costtime_df)  
print("stdtrain_seq2seq_40epoch_costtime_seed_0_df:\n",stdtrain_seq2seq_40epoch_costtime_seed_0_df)  
print("stdtrain_seq2seq_40epoch_costtime_seed_1_df:\n",stdtrain_seq2seq_40epoch_costtime_seed_1_df)  
print("stdtrain_seq2seq_40epoch_costtime_seed_2_df:\n",stdtrain_seq2seq_40epoch_costtime_seed_2_df)  



stdtrain_seq2seq_40epoch_costtime_merge_df = pd.concat([stdtrain_seq2seq_40epoch_costtime_seed_0_df, stdtrain_seq2seq_40epoch_costtime_seed_1_df, stdtrain_seq2seq_40epoch_costtime_seed_2_df])
# stdtrain_seq2seq_40epoch_costtime_merge_df = pd.concat([stdtrain_seq2seq_40epoch_costtime_df, stdtrain_seq2seq_40epoch_costtime_df, stdtrain_seq2seq_40epoch_costtime_df])

print("stdtrain_seq2seq_40epoch_costtime_merge_df:\n",stdtrain_seq2seq_40epoch_costtime_merge_df)  

stdtrain_seq2seq_40epoch_costtime_merge_df = stdtrain_seq2seq_40epoch_costtime_merge_df.reset_index(drop=True)

print("stdtrain_seq2seq_40epoch_costtime_merge_df:\n",stdtrain_seq2seq_40epoch_costtime_merge_df)  


# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


sns_plot=sns.lineplot(data=stdtrain_seq2seq_40epoch_costtime_merge_df, x='epoch', y='costtime', color='b')
ax.set_xlabel('Training Epoch', fontsize=12)
ax.set_ylabel('Cost Time (seconds)', fontsize=12)
ax.set_title('Standard Train Sequence Analyzer', fontsize=14); 
plt.ylim(11, 15)
# plt.ylim(-5, 105)
# 调整横坐标显示范围
# plt.xlim(0, 40)
# 调整横坐标显示范围
# plt.xlim(1, 40)

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.6/training-time/seq2sAna'
savename = f'line-shadow-figure-stdtrain-seq2seq-40epoch-costtime-seed012'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   
