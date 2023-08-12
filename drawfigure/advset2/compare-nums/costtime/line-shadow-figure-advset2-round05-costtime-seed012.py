import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

advset2_rounds05_costtime_on_advmal_file = pd.read_csv('cost_time_list_advset2_round5_seed012.csv')
advset2_rounds05_costtime_on_advmal_df = pd.DataFrame(advset2_rounds05_costtime_on_advmal_file)

advset2_rounds05_costtime_on_advmal_seed_0_df = advset2_rounds05_costtime_on_advmal_df.drop(columns=['seed1','seed2'])  
advset2_rounds05_costtime_on_advmal_seed_0_df.rename(columns={'seed0': 'costtime'}, inplace=True)

advset2_rounds05_costtime_on_advmal_seed_1_df = advset2_rounds05_costtime_on_advmal_df.drop(columns=['seed0','seed2'])  
advset2_rounds05_costtime_on_advmal_seed_1_df.rename(columns={'seed1': 'costtime'}, inplace=True)

advset2_rounds05_costtime_on_advmal_seed_2_df = advset2_rounds05_costtime_on_advmal_df.drop(columns=['seed0','seed1'])  
advset2_rounds05_costtime_on_advmal_seed_2_df.rename(columns={'seed2': 'costtime'}, inplace=True)

print("advset2_rounds05_costtime_on_advmal_df:\n",advset2_rounds05_costtime_on_advmal_df)  
print("advset2_rounds05_costtime_on_advmal_seed_0_df:\n",advset2_rounds05_costtime_on_advmal_seed_0_df)  
print("advset2_rounds05_costtime_on_advmal_seed_1_df:\n",advset2_rounds05_costtime_on_advmal_seed_1_df)  
print("advset2_rounds05_costtime_on_advmal_seed_2_df:\n",advset2_rounds05_costtime_on_advmal_seed_2_df)  



advset2_rounds05_costtime_on_advmal_merge_df = pd.concat([advset2_rounds05_costtime_on_advmal_seed_0_df, advset2_rounds05_costtime_on_advmal_seed_1_df, advset2_rounds05_costtime_on_advmal_seed_2_df])
# advset2_rounds05_costtime_on_advmal_merge_df = pd.concat([advset2_rounds05_costtime_on_advmal_df, advset2_rounds05_costtime_on_advmal_df, advset2_rounds05_costtime_on_advmal_df])

print("advset2_rounds05_costtime_on_advmal_merge_df:\n",advset2_rounds05_costtime_on_advmal_merge_df)  

advset2_rounds05_costtime_on_advmal_merge_df = advset2_rounds05_costtime_on_advmal_merge_df.reset_index(drop=True)

print("advset2_rounds05_costtime_on_advmal_merge_df:\n",advset2_rounds05_costtime_on_advmal_merge_df)  


# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


sns_plot=sns.lineplot(data=advset2_rounds05_costtime_on_advmal_merge_df, x='round', y='costtime', color='b')
ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_ylabel('Cost Time (seconds)', fontsize=12)
ax.set_title('Adversarial Setting II', fontsize=14); 
plt.ylim(-5, 105)

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset2/compare-nums/costtime'
savename = f'line-shadow-figure-advset2-round05-costtime-seed012'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#-----------------------------------------------------------

advset2_rounds10_costtime_on_advmal_file = pd.read_csv('cost_time_list_advset2_round10_seed012.csv')
advset2_rounds10_costtime_on_advmal_df = pd.DataFrame(advset2_rounds10_costtime_on_advmal_file)

advset2_rounds10_costtime_on_advmal_seed_0_df = advset2_rounds10_costtime_on_advmal_df.drop(columns=['seed1','seed2'])  
advset2_rounds10_costtime_on_advmal_seed_0_df.rename(columns={'seed0': 'costtime'}, inplace=True)

advset2_rounds10_costtime_on_advmal_seed_1_df = advset2_rounds10_costtime_on_advmal_df.drop(columns=['seed0','seed2'])  
advset2_rounds10_costtime_on_advmal_seed_1_df.rename(columns={'seed1': 'costtime'}, inplace=True)

advset2_rounds10_costtime_on_advmal_seed_2_df = advset2_rounds10_costtime_on_advmal_df.drop(columns=['seed0','seed1'])  
advset2_rounds10_costtime_on_advmal_seed_2_df.rename(columns={'seed2': 'costtime'}, inplace=True)

print("advset2_rounds10_costtime_on_advmal_df:\n",advset2_rounds10_costtime_on_advmal_df)  
print("advset2_rounds10_costtime_on_advmal_seed_0_df:\n",advset2_rounds10_costtime_on_advmal_seed_0_df)  
print("advset2_rounds10_costtime_on_advmal_seed_1_df:\n",advset2_rounds10_costtime_on_advmal_seed_1_df)  
print("advset2_rounds10_costtime_on_advmal_seed_2_df:\n",advset2_rounds10_costtime_on_advmal_seed_2_df)  



advset2_rounds10_costtime_on_advmal_merge_df = pd.concat([advset2_rounds10_costtime_on_advmal_seed_0_df, advset2_rounds10_costtime_on_advmal_seed_1_df, advset2_rounds10_costtime_on_advmal_seed_2_df])
# advset2_rounds10_costtime_on_advmal_merge_df = pd.concat([advset2_rounds10_costtime_on_advmal_df, advset2_rounds10_costtime_on_advmal_df, advset2_rounds10_costtime_on_advmal_df])

print("advset2_rounds10_costtime_on_advmal_merge_df:\n",advset2_rounds10_costtime_on_advmal_merge_df)  

advset2_rounds10_costtime_on_advmal_merge_df = advset2_rounds10_costtime_on_advmal_merge_df.reset_index(drop=True)

print("advset2_rounds10_costtime_on_advmal_merge_df:\n",advset2_rounds10_costtime_on_advmal_merge_df)  


# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


sns_plot=sns.lineplot(data=advset2_rounds10_costtime_on_advmal_merge_df, x='round', y='costtime', color='b')
ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_ylabel('Cost Time (seconds)', fontsize=12)
ax.set_title('Adversarial Setting II', fontsize=14); 
plt.ylim(-5, 105)

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset2/compare-nums/costtime'
savename = f'line-shadow-figure-advset2-round10-costtime-seed012'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#--------------------------
advset2_rounds20_costtime_on_advmal_file = pd.read_csv('cost_time_list_advset2_round20_seed012.csv')
advset2_rounds20_costtime_on_advmal_df = pd.DataFrame(advset2_rounds20_costtime_on_advmal_file)

advset2_rounds20_costtime_on_advmal_seed_0_df = advset2_rounds20_costtime_on_advmal_df.drop(columns=['seed1','seed2'])  
advset2_rounds20_costtime_on_advmal_seed_0_df.rename(columns={'seed0': 'costtime'}, inplace=True)

advset2_rounds20_costtime_on_advmal_seed_1_df = advset2_rounds20_costtime_on_advmal_df.drop(columns=['seed0','seed2'])  
advset2_rounds20_costtime_on_advmal_seed_1_df.rename(columns={'seed1': 'costtime'}, inplace=True)

advset2_rounds20_costtime_on_advmal_seed_2_df = advset2_rounds20_costtime_on_advmal_df.drop(columns=['seed0','seed1'])  
advset2_rounds20_costtime_on_advmal_seed_2_df.rename(columns={'seed2': 'costtime'}, inplace=True)

print("advset2_rounds20_costtime_on_advmal_df:\n",advset2_rounds20_costtime_on_advmal_df)  
print("advset2_rounds20_costtime_on_advmal_seed_0_df:\n",advset2_rounds20_costtime_on_advmal_seed_0_df)  
print("advset2_rounds20_costtime_on_advmal_seed_1_df:\n",advset2_rounds20_costtime_on_advmal_seed_1_df)  
print("advset2_rounds20_costtime_on_advmal_seed_2_df:\n",advset2_rounds20_costtime_on_advmal_seed_2_df)  



advset2_rounds20_costtime_on_advmal_merge_df = pd.concat([advset2_rounds20_costtime_on_advmal_seed_0_df, advset2_rounds20_costtime_on_advmal_seed_1_df, advset2_rounds20_costtime_on_advmal_seed_2_df])
# advset2_rounds20_costtime_on_advmal_merge_df = pd.concat([advset2_rounds20_costtime_on_advmal_df, advset2_rounds20_costtime_on_advmal_df, advset2_rounds20_costtime_on_advmal_df])

print("advset2_rounds20_costtime_on_advmal_merge_df:\n",advset2_rounds20_costtime_on_advmal_merge_df)  

advset2_rounds20_costtime_on_advmal_merge_df = advset2_rounds20_costtime_on_advmal_merge_df.reset_index(drop=True)

print("advset2_rounds20_costtime_on_advmal_merge_df:\n",advset2_rounds20_costtime_on_advmal_merge_df)  


# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


sns_plot=sns.lineplot(data=advset2_rounds20_costtime_on_advmal_merge_df, x='round', y='costtime', color='b')
ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_ylabel('Cost Time (seconds)', fontsize=12)
ax.set_title('Adversarial Setting II', fontsize=14); 
plt.ylim(-5, 105)

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset2/compare-nums/costtime'
savename = f'line-shadow-figure-advset2-round20-costtime-seed012'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   


#----------------------------------------

# sns.set_theme(style="darkgrid")
plt.style.use('seaborn') 
# plt.style.use('default') 
# plt.style.use('seaborn-deep') 

# plt.figure(figsize=(4, 3.5), dpi=500)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率




sns_plot=sns.lineplot(data=advset2_rounds20_costtime_on_advmal_merge_df, x='round', y='costtime', color='b', label='Round=20, AdvSet Size=15')



sns_plot=sns.lineplot(data=advset2_rounds10_costtime_on_advmal_merge_df, x='round', y='costtime', color='r', label='Round=10, AdvSet Size=28')


sns_plot=sns.lineplot(data=advset2_rounds05_costtime_on_advmal_merge_df, x='round', y='costtime', color='g', label='Round=05, AdvSet Size=53')

ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_ylabel('Cost Time (seconds)', fontsize=12)
ax.set_title('Adversarial Setting II', fontsize=14); 
plt.ylim(-5, 105)
plt.legend(loc='lower right')
plt.xlim(0, 5)

savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset2/compare-nums/costtime'
savename = f'line-shadow-figure-advset2-round051020-costtime-seed012'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   