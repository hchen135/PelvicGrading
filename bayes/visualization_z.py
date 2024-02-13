import pandas as pd 
import numpy as np 
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

df = pd.read_csv("all_results_graph1_visualization_zlow_change.csv")

GT_R_AUC = df.loc[0].at['GT_ROT_auc']
GT_T_AUC = df.loc[0].at['GT_TRA_auc']

HIGH_R_AUC = df.loc[len(df)-1].at['High_ROT_auc']
HIGH_T_AUC = df.loc[len(df)-1].at['High_TRA_auc']

desired_df = df[df['High_threshold'] == 0.8]

LOW_threshold = np.array(desired_df.loc[:,'Low_threhsold'])
LOW_R_AUC = np.array(desired_df.loc[:,'Low_ROT_auc'])
LOW_T_AUC = np.array(desired_df.loc[:,'Low_TRA_auc'])
REFINE_R_AUC = np.array(desired_df.loc[:,'Refine_ROT_auc'])
REFINE_T_AUC = np.array(desired_df.loc[:,'Refine_TRA_auc'])

print(LOW_threshold)
# smooth
X_ = np.linspace(LOW_threshold.min(), LOW_threshold.max(), 500)

X_Y_Spline = make_interp_spline(LOW_threshold, LOW_R_AUC)
LOW_R_AUC = X_Y_Spline(X_)

X_Y_Spline = make_interp_spline(LOW_threshold, LOW_T_AUC)
LOW_T_AUC = X_Y_Spline(X_)

X_Y_Spline = make_interp_spline(LOW_threshold, REFINE_R_AUC)
REFINE_R_AUC = X_Y_Spline(X_)

X_Y_Spline = make_interp_spline(LOW_threshold, REFINE_T_AUC)
REFINE_T_AUC = X_Y_Spline(X_)

fig = plt.gcf()
fig.set_size_inches(12, 4)
ax = plt.subplot(111)

ax.plot(X_, LOW_R_AUC,'b--',label='LOW_R',alpha=0.2)
ax.plot(X_, LOW_T_AUC,'g--',label='LOW_T',alpha=0.2)
ax.plot([0.1,0.8], [HIGH_R_AUC,HIGH_R_AUC],'b-',label='HIGH_R')
ax.plot([0.1,0.8], [HIGH_T_AUC,HIGH_T_AUC],'g-',label='HIGH_T')
ax.plot(X_, REFINE_R_AUC,'b--',label='REFINE_R')
ax.plot(X_, REFINE_T_AUC,'g--',label='REFINE_T')

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

plt.title('Impact of the magnitude of z_low for BM refinement')
plt.xlabel('z_low')
plt.ylabel('AUC')
plt.xlim(0.1,0.8)
plt.ylim(0.64,0.91)
# ax.legend()
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=6)

plt.savefig("zlow_behavior.png")

plt.clf()






















df = pd.read_csv("all_results_graph1_visualization_z*_change.csv")

GT_R_AUC = df.loc[0].at['GT_ROT_auc']
GT_T_AUC = df.loc[0].at['GT_TRA_auc']

HIGH_R_AUC = df.loc[len(df)-1].at['High_ROT_auc']
HIGH_T_AUC = df.loc[len(df)-1].at['High_TRA_auc']

desired_df = df[df['High_threshold'] == 0.8]

threshold = np.array(desired_df.loc[:,'Threshold'])
LOW_R_AUC = np.array(desired_df.loc[:,'Low_ROT_auc'])
LOW_T_AUC = np.array(desired_df.loc[:,'Low_TRA_auc'])
REFINE_R_AUC = np.array(desired_df.loc[:,'Refine_ROT_auc'])
REFINE_T_AUC = np.array(desired_df.loc[:,'Refine_TRA_auc'])

print(threshold)
# smooth
X_ = np.linspace(threshold.min(), threshold.max(), 500)

X_Y_Spline = make_interp_spline(threshold, LOW_R_AUC)
LOW_R_AUC = X_Y_Spline(X_)

X_Y_Spline = make_interp_spline(threshold, LOW_T_AUC)
LOW_T_AUC = X_Y_Spline(X_)

X_Y_Spline = make_interp_spline(threshold, REFINE_R_AUC)
REFINE_R_AUC = X_Y_Spline(X_)

X_Y_Spline = make_interp_spline(threshold, REFINE_T_AUC)
REFINE_T_AUC = X_Y_Spline(X_)

fig = plt.gcf()
fig.set_size_inches(12, 4)
ax = plt.subplot(111)

ax.plot(X_, LOW_R_AUC,'b--',label='LOW_R',alpha=0.2)
ax.plot(X_, LOW_T_AUC,'g--',label='LOW_T',alpha=0.2)
ax.plot([0.1,0.9], [HIGH_R_AUC,HIGH_R_AUC],'b-',label='HIGH_R')
ax.plot([0.1,0.9], [HIGH_T_AUC,HIGH_T_AUC],'g-',label='HIGH_T')
ax.plot(X_, REFINE_R_AUC,'b--',label='REFINE_R')
ax.plot(X_, REFINE_T_AUC,'g--',label='REFINE_T')



box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# plt.plot([0.3,0.8], [GT_R_AUC,GT_R_AUC],'b-')
# plt.plot([0.3,0.8], [GT_T_AUC,GT_T_AUC],'g-')

plt.title('Impact of the magnitude of z* for BM refinement')
plt.xlabel('z*')
plt.ylabel('AUC')
plt.xlim(0.1,0.8)
plt.ylim(0.64,0.91)
# ax.legend()
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=6)

plt.savefig("zstar_behavior.png")

# print(desired_df)
