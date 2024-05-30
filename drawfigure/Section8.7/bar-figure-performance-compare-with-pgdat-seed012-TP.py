import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

performance_compare_with_pgdat_seed012_file = pd.read_csv('performance-compare-with-pgdat-seed012.csv')
performance_compare_with_pgdat_seed012_df = pd.DataFrame(performance_compare_with_pgdat_seed012_file)
print("performance_compare_with_pgdat_seed012_df:\n",performance_compare_with_pgdat_seed012_df)  

""" 
'Adv-TP','Adv-FN','Adv-FNR','Adv-Recall',
'Cle-TP','Cle-FN','Cle-TN','Cle-FP',
'Cle-FNR','Cle-FPR',
'Cle-Precision','Cle-Recall','Cle-F1','Cle-Accuracy'
"""




#============================ Adv TP ============================
performance_compare_with_pgdat_seed012_adv_TP_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-FN','Adv-FNR','Adv-Recall',
    'Cle-TP','Cle-FN','Cle-TN','Cle-FP',
    'Cle-FNR','Cle-FPR',
    'Cle-Precision','Cle-Recall','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_adv_TP_df:\n",performance_compare_with_pgdat_seed012_adv_TP_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_adv_TP_df, x='Seed', y='Adv-TP', hue="Method", palette="Set1")
# """ 
# [
# '#4c72b0',蓝
# '#55a868', 绿
# '#c44e52', 红
# '#8172b2',紫 
# '#ccb974',
# '#64b5cd'
# ]
# """
# palette=["#55a868","#4c72b0","#c44e52"]

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('TP Value', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='center right',frameon=True)

ax.set_title('Performance on Adversarial Samples', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-advTP'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   
#================================================================




#============================ Adv FN ============================
performance_compare_with_pgdat_seed012_adv_FN_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FNR','Adv-Recall',
    'Cle-TP','Cle-FN','Cle-TN','Cle-FP',
    'Cle-FNR','Cle-FPR',
    'Cle-Precision','Cle-Recall','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_adv_FN_df:\n",performance_compare_with_pgdat_seed012_adv_FN_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_adv_FN_df, x='Seed', y='Adv-FN', hue="Method", palette="Set1")

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('FN Value', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='lower right',frameon=True)
plt.ylim(0, 100)

ax.set_title('Performance on Adversarial Samples', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-advFN'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================



#============================ Adv FNR ============================
performance_compare_with_pgdat_seed012_adv_FNR_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-Recall',
    'Cle-TP','Cle-FN','Cle-TN','Cle-FP',
    'Cle-FNR','Cle-FPR',
    'Cle-Precision','Cle-Recall','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_adv_FNR_df:\n",performance_compare_with_pgdat_seed012_adv_FNR_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_adv_FNR_df, x='Seed', y='Adv-FNR', hue="Method", palette="Set1")

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('FNR Value (%)', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='best')
plt.ylim(0, 100)

ax.set_title('Performance on Adversarial Samples', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-advFNR'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================


#============================ Adv Recall ============================
performance_compare_with_pgdat_seed012_adv_Recall_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-FNR',
    'Cle-TP','Cle-FN','Cle-TN','Cle-FP',
    'Cle-FNR','Cle-FPR',
    'Cle-Precision','Cle-Recall','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_adv_Recall_df:\n",performance_compare_with_pgdat_seed012_adv_Recall_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_adv_Recall_df, x='Seed', y='Adv-Recall', hue="Method", palette="Set1")

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('Recall Value (%)', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='center right',frameon=True)
plt.ylim(0, 100)

ax.set_title('Performance on Adversarial Samples', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-advRecall'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================















#........................................................................................................
colors = sns.color_palette("tab10", 2)
#============================ cle TP ============================
performance_compare_with_pgdat_seed012_cle_TP_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-FNR','Adv-Recall',
    'Cle-FN','Cle-TN','Cle-FP',
    'Cle-FNR','Cle-FPR',
    'Cle-Precision','Cle-Recall','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_cle_TP_df:\n",performance_compare_with_pgdat_seed012_cle_TP_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_cle_TP_df, x='Seed', y='Cle-TP', hue="Method", palette=[colors[1],colors[0]])

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('TP Value', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='center right',frameon=True)
plt.ylim(0, 350)

ax.set_title('Performance on Clean Test Set', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-cleTP'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================


#============================ cle FN ============================
performance_compare_with_pgdat_seed012_cle_FN_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-FNR','Adv-Recall',
    'Cle-TP','Cle-TN','Cle-FP',
    'Cle-FNR','Cle-FPR',
    'Cle-Precision','Cle-Recall','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_cle_FN_df:\n",performance_compare_with_pgdat_seed012_cle_FN_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_cle_FN_df, x='Seed', y='Cle-FN', hue="Method", palette=[colors[1],colors[0]])

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('FN Value', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='best')
plt.ylim(0, 100)

ax.set_title('Performance on Clean Test Set', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-cleFN'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================




#============================ cle TN ============================
performance_compare_with_pgdat_seed012_cle_TN_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-FNR','Adv-Recall',
    'Cle-TP','Cle-FN','Cle-FP',
    'Cle-FNR','Cle-FPR',
    'Cle-Precision','Cle-Recall','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_cle_TN_df:\n",performance_compare_with_pgdat_seed012_cle_TN_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_cle_TN_df, x='Seed', y='Cle-TN', hue="Method", palette=[colors[1],colors[0]])

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('TN Value', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='center right',frameon=True)
plt.ylim(0, 4000)

ax.set_title('Performance on Clean Test Set', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-cleTN'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================




#============================ cle FP ============================
performance_compare_with_pgdat_seed012_cle_FP_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-FNR','Adv-Recall',
    'Cle-TP','Cle-FN','Cle-TN',
    'Cle-FNR','Cle-FPR',
    'Cle-Precision','Cle-Recall','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_cle_FP_df:\n",performance_compare_with_pgdat_seed012_cle_FP_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_cle_FP_df, x='Seed', y='Cle-FP', hue="Method", palette=[colors[1],colors[0]])

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('FP Value', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='center right',frameon=True)
# plt.ylim(0, 4000)

ax.set_title('Performance on Clean Test Set', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-cleFP'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================



#============================ cle FNR ============================
performance_compare_with_pgdat_seed012_cle_FNR_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-FNR','Adv-Recall',
    'Cle-TP','Cle-FN','Cle-TN','Cle-FP',
    'Cle-FPR',
    'Cle-Precision','Cle-Recall','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_cle_FNR_df:\n",performance_compare_with_pgdat_seed012_cle_FNR_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_cle_FNR_df, x='Seed', y='Cle-FNR', hue="Method", palette=[colors[1],colors[0]])

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('FNR Value (%)', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='upper right')
plt.ylim(0, 100)

ax.set_title('Performance on Clean Test Set', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-cleFNR'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================



#============================ cle FPR ============================
performance_compare_with_pgdat_seed012_cle_FPR_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-FNR','Adv-Recall',
    'Cle-TP','Cle-FN','Cle-TN', 'Cle-FP',
    'Cle-FNR',
    'Cle-Precision','Cle-Recall','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_cle_FPR_df:\n",performance_compare_with_pgdat_seed012_cle_FPR_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_cle_FPR_df, x='Seed', y='Cle-FPR', hue="Method", palette=[colors[1],colors[0]])

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('FPR Value (%)', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='upper right')
plt.ylim(0, 100)

ax.set_title('Performance on Clean Test Set', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-cleFPR'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================




#============================ cle Precision ============================
performance_compare_with_pgdat_seed012_cle_Precision_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-FNR','Adv-Recall',
    'Cle-TP','Cle-FN','Cle-TN', 'Cle-FP',
    'Cle-FNR','Cle-FPR',
    'Cle-Recall','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_cle_Precision_df:\n",performance_compare_with_pgdat_seed012_cle_Precision_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_cle_Precision_df, x='Seed', y='Cle-Precision', hue="Method", palette=[colors[1],colors[0]])

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('Precision Value (%)', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='upper right')
plt.ylim(0, 100)

ax.set_title('Performance on Clean Test Set', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-clePrecision'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================




#============================ cle Recall ============================
performance_compare_with_pgdat_seed012_cle_Recall_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-FNR','Adv-Recall',
    'Cle-TP','Cle-FN','Cle-TN', 'Cle-FP',
    'Cle-FNR','Cle-FPR',
    'Cle-Precision','Cle-F1','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_cle_Recall_df:\n",performance_compare_with_pgdat_seed012_cle_Recall_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_cle_Recall_df, x='Seed', y='Cle-Recall', hue="Method", palette=[colors[1],colors[0]])

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('Recall Value (%)', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='center right',frameon=True)
plt.ylim(0, 100)

ax.set_title('Performance on Clean Test Set', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-cleRecall'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================



#============================ cle F1 ============================
performance_compare_with_pgdat_seed012_cle_F1_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-FNR','Adv-Recall',
    'Cle-TP','Cle-FN','Cle-TN', 'Cle-FP',
    'Cle-FNR','Cle-FPR',
    'Cle-Precision','Cle-Recall','Cle-Accuracy'
    ])  

print("performance_compare_with_pgdat_seed012_cle_F1_df:\n",performance_compare_with_pgdat_seed012_cle_F1_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率

sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_cle_F1_df, x='Seed', y='Cle-F1', hue="Method", palette=[colors[1],colors[0]])

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('F1 Value (%)', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='upper right')
plt.ylim(0, 100)

ax.set_title('Performance on Clean Test Set', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-cleF1'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   

#================================================================



#============================ cle Accuracy ============================
performance_compare_with_pgdat_seed012_cle_Accuracy_df = performance_compare_with_pgdat_seed012_df.drop(columns=[
    'Adv-TP','Adv-FN','Adv-FNR','Adv-Recall',
    'Cle-TP','Cle-FN','Cle-TN', 'Cle-FP',
    'Cle-FNR','Cle-FPR',
    'Cle-Precision','Cle-Recall','Cle-F1'
    ])  

print("performance_compare_with_pgdat_seed012_cle_Accuracy_df:\n",performance_compare_with_pgdat_seed012_cle_Accuracy_df)  

plt.style.use('seaborn') 
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) #dpi=100 分辨率


# colors = sns.color_palette("Set1", 4)
sns_plot=sns.barplot(data=performance_compare_with_pgdat_seed012_cle_Accuracy_df, x='Seed', y='Cle-Accuracy', hue="Method", palette=[colors[1],colors[0]])

""" 
[
'#4c72b0',蓝
'#55a868', 绿
'#c44e52', 红
'#8172b2',紫 
'#ccb974',
'#64b5cd'
]
"""

# ax.set_xlabel('Evolution Round', fontsize=12)
ax.set_xlabel('') 
ax.set_ylabel('Accuracy Value (%)', fontsize=12)
plt.legend(title=None)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc='center right',frameon=True)
plt.ylim(0, 100)

ax.set_title('Performance on Clean Test Set', fontsize=14); 

savepath = f'/home/huan1932/ARIoTEDef-Actual/drawfigure/Section8.7'
savename = f'bar-figure-performance-compare-with-pgdat-seed012-cleAccuracy'
plt.savefig(fname=f'{savepath}/{savename}.png',dpi=500)
plt.close   


#================================================================