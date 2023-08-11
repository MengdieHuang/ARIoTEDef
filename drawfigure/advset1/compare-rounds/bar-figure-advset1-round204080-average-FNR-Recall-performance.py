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


advset1_rounds204080_TPFN_on_advma_merge_df = pd.concat([advset1_rounds204080_TP_on_advmal_df, advset1_rounds204080_FN_on_advmal_df])


advset1_rounds204080_FNR_on_advmal_df = advset1_rounds204080_on_advmal_file.drop(columns=['TP','FN', 'Recall'])  
advset1_rounds204080_FNR_on_advmal_df.rename(columns={'FNR': 'Value'}, inplace=True)
advset1_rounds204080_FNR_on_advmal_df['FNR_Recall_label']='FNR'    

advset1_rounds204080_Recall_on_advmal_df = advset1_rounds204080_on_advmal_file.drop(columns=['TP','FN', 'FNR'])  
advset1_rounds204080_Recall_on_advmal_df.rename(columns={'Recall': 'Value'}, inplace=True)
advset1_rounds204080_Recall_on_advmal_df['FNR_Recall_label']='Recall'    

print("advset1_rounds204080_FNR_on_advmal_df:\n",advset1_rounds204080_FNR_on_advmal_df)  
print("advset1_rounds204080_Recall_on_advmal_df:\n",advset1_rounds204080_Recall_on_advmal_df)  

advset1_rounds204080_FNRRecall_on_advma_merge_df = pd.concat([advset1_rounds204080_Recall_on_advmal_df,advset1_rounds204080_FNR_on_advmal_df])


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
sns_plot=sns.barplot(data=advset1_rounds204080_FNRRecall_on_advma_merge_df, x='round', y='Value', hue="FNR_Recall_label", palette="tab10")

# sns_plot=sns.barplot(data=advset1_rounds204080_on_advmal_df, x='round', y='FN', label='FN', color='o')

# sns.barplot(data=df, x="island", y="body_mass_g", hue="TP_FN_label")


ax.set_xlabel('Evolved Model', fontsize=12)
ax.set_ylabel('FNR Value / Recall Value (%)', fontsize=12)
# plt.legend(
#     fontsize=8, 
#     loc='upper right') 
plt.legend(title=None)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax.set_title('Adversarial Setting I', fontsize=14); 

# 移除默认的 Seaborn 图例
# # 创建一个新的坐标轴用于放置图例
# ax_legend = ax.figure.add_axes([0.8, 0.8, 0.1, 0.1])  # 调整 [x, y, width, height] 控制图例的位置和大小

# # 在新坐标轴上添加图例
# sns.barplot(x=[0], y=[0], color='blue', ax=ax_legend, label='Legend Label')  # 这里只是添加一个示例图例

# ax_legend.set_xlabel('')  # 清除新坐标轴的 x 标签
# ax_legend.set_ylabel('')  # 清除新坐标轴的 y 标签


savepath = f'/home/huan1932/ARIoTEDef/drawfigure/advset1/compare-rounds'
savename = f'bar-figure-advset1-round204080-average-FNR-Recall-performance'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   
