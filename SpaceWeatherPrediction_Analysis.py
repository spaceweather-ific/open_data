import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
#sns.set_style('dark')
plt.rcParams['font.size'] = 16
import json
import os
import copy
from IPython.display import display, HTML

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

################# Fetching data #################

base_dir = 'predictionsAndTest_values/'
srt = ['bs','do']
#st = 10
names = [['predict_storm{}_withBootStrapLSTM.csv'.format(st),'predict_storm{}_withDropoutLSTM.csv'.format(st)] for st in range(17)]#,'predict_storm1_withDropoutBig.csv']
DF = [{} for i in range(17)]
TF = [{} for i in range(17)]

for st in range(17):
  for i,j in enumerate(srt):
    print(names[st][i])
    DF[st][j] = pd.read_csv(base_dir+names[st][i], header = None, names = ['s','t','n','sym'])
    if i == 1:
      dropout_suffix = 'Dropout'
    else:
      dropout_suffix = ''
    # now the test values, these are the ones produced with the dropout runs, shouldn't be different
    TF[st][j] = pd.read_csv(base_dir+'test_storm{}{}.csv'.format(st,dropout_suffix), header = None, names = ['s','t','sym'])

n_syst  = [int(DF[st][srt[0]]['n'].iloc[-1]+1) for st in range(17)]
n_syst_do  = [int(DF[st][srt[1]]['n'].iloc[-1]+1) for st in range(17)]

t_range = [[[int(DF[st]['bs']['t'].iloc[0]),int(DF[st]['bs']['t'].iloc[-1])+1] for st in range(17)],
               [[int(DF[st]['do']['t'].iloc[0]),int(DF[st]['do']['t'].iloc[-1])+1] for st in range(17)]]
shifts = [t_range[1][st][1]-t_range[0][st][1] for st in range(17)]

################# Predictions Visualized #################

#~~~~~~~~~~~~~~~~~ Producing means of predictions ~~~~~~~~~~~~~~~~~#
means = [{} for st in range(17)]
stds = [{} for st in range(17)]

for st in range(17):
  for j,i in enumerate(srt):
    print(st,i,t_range[j][st][1]-t_range[j][st][0])
    means[st][i] = np.zeros(t_range[j][st][1]-t_range[j][st][0])
    stds[st][i] = np.zeros(t_range[j][st][1]-t_range[j][st][0])
    
    for t in range(t_range[j][st][0],t_range[j][st][1]):
      df = DF[st][i][DF[st][i]['t']==t]
      means[st][i][t] = np.mean(df['sym'])
      stds[st][i][t] = np.std(df['sym'])

# the width of the error bar (95% confidence interval for sigma = 2)
sigma_n = 2

# values used for standardization in the original code, used to convert to nT
sym_std, sym_mean = 42.573086, -34.011437

srt_name = ['bootstrap','dropout']

#~~~~~~~~~~~~~~~~~ Defining limits of visualization and analysis ~~~~~~~~~~~~~~~~~#
# Saying peaks = True means restricting visualization and analysis only to the defined limits of the peak.

peaks = False
peaks_peaks = False
base_dir = '/content/gdrive/My Drive/SpaceWeather_OutputAnalysis/'

if peaks:
  # Criteria for peaks: by eye. only one peak, the biggest one. As long as the interval includes width at half max
  peaks_loc_steps_bs = [[190, 300, 230, 180, 340, 230, 250, 260, 180, 120, 210, 180, 1110, 200, 100, 320, 170],
                        [250, 360, 330, 250, 440, 330, 340, 350, 300, 280, 360, 280, 1210, 300, 250, 420, 300]]
  if peaks_peaks:
    peaks_loc_steps_bs = [[5480//25, 8500//25, 6200//25, 5530//25, 9250//25, 6480//25, 6800//25, 7000//25, 4500//25, 4150//25, 5900//25, 180, 1110, 200, 100, 9450//25, 170],
                          [5750//25, 8730//25, 6600//25, 5740//25, 9600//25, 6800//25, 7000//25, 7400//25, 5300//25, 4800//25, 6400//25, 280, 1210, 300, 250, 9700//25, 300]]
                          
  a = np.array([[5*peaks_loc_steps_bs[0][st]            for st in range(17)],
                [5*peaks_loc_steps_bs[0][st]+35 for st in range(17)]]) # gotta introduce the shift, peak locs were chosen from bootstrap values
  b = np.array([[5*peaks_loc_steps_bs[1][st]            for st in range(17)],
                [5*peaks_loc_steps_bs[1][st]+35 for st in range(17)]])
  peak_prefix = 'PEAK_'
  base_dir = base_dir+"peaks_bsdo/"
else:
  a = [[t_range[0][st][0] for st in range(17)],[t_range[1][st][0] for st in range(17)]]
  b = [[t_range[0][st][1] for st in range(17)],[t_range[1][st][1] for st in range(17)]]
  peak_prefix = ''
  base_dir = base_dir+'bsdo/'

#~~~~~~~~~~~~~~~~~ Main body plot ~~~~~~~~~~~~~~~~~#
fig, axes = plt.subplots(len(srt),2, gridspec_kw={'height_ratios': [2, 1]}, figsize = (7.5*len(srt),9))
fig.subplots_adjust(wspace=0.2, hspace = 0)

st1, st2 = 1, 11

for st_i, st in enumerate([st1,st2]):
  for ratio_n in range(2):
    for i,j in enumerate([srt[0]]):
      if ratio_n == 0:
        axes[ratio_n][st_i].set_title('Storm T{}'.format(st+1))

        y_mean = (means[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
        y_min = (means[st][j][a[i][st]:b[i][st]] - sigma_n*stds[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
        y_max = (means[st][j][a[i][st]:b[i][st]] + sigma_n*stds[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
        test_values = TF[st][j]['sym'][a[i][st]:b[i][st]]*sym_std+sym_mean
        y_range = max(test_values) - min(test_values)

        if i == 0:
          y_lim_1, y_lim_2 = min(y_min[10:])-0.1*y_range, max(y_max[10:])+0.1*y_range

        t_axis = 5*np.array(range(b[i][st]-a[i][st]))
        axes[ratio_n][st_i].plot(t_axis, y_mean, '--', label = 'Prediction mean', color = 'red')
        axes[ratio_n][st_i].fill_between(t_axis, y_min, y_max,
                        color = 'orange', alpha = 0.7, label = 'Prediction (95% CL)')
                
        axes[ratio_n][st_i].plot(t_axis, test_values, alpha = 1, label = 'Test values')
        
        if i == 0:
          axes[ratio_n][st_i].set_ylabel('SYM-H (nT)')
        else:
          axes[ratio_n][st_i].set_yticks([])
        #axes[ratio_n][i].set_xlabel('Time (min)')
        axes[ratio_n][st_i].set_xticks([])
        axes[ratio_n][st_i].set_ylim(y_lim_1, y_lim_2)
        axes[ratio_n][st_i].set_xlim(5*a[i][st], 5*b[i][st])

        handles, labels = axes[ratio_n][st_i].get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        axes[ratio_n][st_i].legend(handles, labels, loc = 3, fontsize = 12)
        #axes[ratio_n][st].grid(True)

      if ratio_n == 1:
        #axes[ratio_n][st].set_title('Storm T{}, '.format(st+1)+srt_name[i])

        y_mean = (means[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
        y_min = (means[st][j][a[i][st]:b[i][st]] - sigma_n*stds[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
        y_max = (means[st][j][a[i][st]:b[i][st]] + sigma_n*stds[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
        test_values = TF[st][j]['sym'][a[i][st]:b[i][st]]*sym_std+sym_mean
        y_range = max(test_values) - min(test_values)

        if i == 0:
          y_lim_1, y_lim_2 = min(y_min[10:])-0.1*y_range, max(y_max[10:])+0.1*y_range

        t_axis = 5*np.array(range(b[i][st]-a[i][st]))
        rmse = 1#np.sqrt(mean_squared_error(y_mean,test_values))
        axes[ratio_n][st_i].plot(t_axis, (y_mean-y_mean)/rmse, '--', label = 'Prediction mean', color = 'red')
        axes[ratio_n][st_i].fill_between(t_axis, (y_min-y_mean)/rmse, (y_max-y_mean)/rmse,
                        color = 'orange', alpha = 0.7, label = 'Prediction (95% CL)')
                
        axes[ratio_n][st_i].plot(t_axis, (test_values-y_mean)/rmse, alpha = 1, label = 'Test values')

        if i == 0:
          axes[ratio_n][st_i].set_ylabel('Data-Model (nT)')
        else:
          axes[ratio_n][st_i].set_yticks([])
        #axes[ratio_n][st].set_xlabel('Time (min)')
        y_scale = max(max(abs((test_values-y_mean)/rmse)),max(abs((y_max-y_mean)/rmse)))*1.1 #whichever is bigger between orange band and blue line
        axes[ratio_n][st_i].set_ylim(-y_scale,y_scale)
        axes[ratio_n][st_i].set_xlim(5*a[i][st], 5*b[i][st])
        #print(ratio_n,st,i)
        axes[ratio_n][st_i].set_xlabel('time (min)')

        handles, labels = axes[ratio_n][st_i].get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        #axes[ratio_n][st_i].legend(handles, labels, loc = 3, fontsize = 11)
        #axes[ratio_n][st].grid(True)

pic_name = 'ACEsystematics1_predLSTM200_withBootStrap(2)_{}&{}.pdf'.format(st1+1,st2+1)
plt.savefig(base_dir+pic_name, format = 'pdf')

#~~~~~~~~~~~~~~~~~ Final bootstrapVSdropout plots ~~~~~~~~~~~~~~~~~#
# The plot cell should be run over and over to produce 16 pdf. The for loop + clear function in matplot lib used to work, but it stopped working after adding the lower panels for some reasons and didn't troubleshoot further

st = 0
fig, axes = plt.subplots(len(srt),2, gridspec_kw={'height_ratios': [2, 1]}, figsize = (7.5*len(srt),9))
fig.subplots_adjust(wspace=0, hspace = 0)
for ratio_n in range(2):
  for i,j in enumerate(srt):
    if ratio_n == 0:
      #axes[ratio_n][i].subplots_adjust(wspace=0)
      axes[ratio_n][i].set_title('Storm T{}, '.format(st+1)+srt_name[i])

      y_mean = (means[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
      y_min = (means[st][j][a[i][st]:b[i][st]] - sigma_n*stds[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
      y_max = (means[st][j][a[i][st]:b[i][st]] + sigma_n*stds[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
      test_values = TF[st][j]['sym'][a[i][st]:b[i][st]]*sym_std+sym_mean
      y_range = max(test_values) - min(test_values)

      if i == 0:
        y_lim_1, y_lim_2 = min(test_values)-0.1*y_range, max(test_values)+0.1*y_range

      t_axis = 5*np.array(range(a[i][st],b[i][st]))
      axes[ratio_n][i].plot(t_axis, y_mean, '--', label = 'Prediction mean', color = 'red')
      axes[ratio_n][i].fill_between(t_axis, y_min, y_max,
                                    color = 'orange', alpha = 0.7, label = 'Prediction (95% CL)')
                
      axes[ratio_n][i].plot(t_axis, test_values, alpha = 1, label = 'Test values')
      
      if i == 0:
        axes[ratio_n][i].set_ylabel('SYM-H (nT)')
      else:
        axes[ratio_n][i].set_yticks([])
      #axes[ratio_n][i].set_xlabel('Time (min)')
      #axes[ratio_n][i].set_xticks([])
      #axes[ratio_n][i].grid()
      axes[ratio_n][i].set_ylim(y_lim_1, y_lim_2)
      axes[ratio_n][i].set_xlim(5*a[i][st], 5*b[i][st])

      handles, labels = axes[ratio_n][i].get_legend_handles_labels()
      # sort both labels and handles by labels
      labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
      axes[ratio_n][i].legend(handles, labels, fontsize = 12)
      #axes[ratio_n][i].grid(True)

    if ratio_n == 1:
      #axes[ratio_n][i].subplots_adjust(wspace=0)
      #axes[ratio_n][i].set_title('Storm T{}, '.format(st+1)+srt_name[i])

      y_mean = (means[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
      y_min = (means[st][j][a[i][st]:b[i][st]] - sigma_n*stds[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
      y_max = (means[st][j][a[i][st]:b[i][st]] + sigma_n*stds[st][j][a[i][st]:b[i][st]])*sym_std+sym_mean
      test_values = TF[st][j]['sym'][a[i][st]:b[i][st]]*sym_std+sym_mean

      t_axis = 5*np.array(range(a[i][st],b[i][st]))
      y_mean = np.nan_to_num(y_mean)
      rmse = 1#np.sqrt(mean_squared_error(y_mean,test_values))
      axes[ratio_n][i].plot(t_axis, (y_mean-y_mean)/rmse, '--', label = 'Prediction mean', color = 'red')
      axes[ratio_n][i].fill_between(t_axis, (y_min-y_mean)/rmse, (y_max-y_mean)/rmse,
                                    color = 'orange', alpha = 0.7, label = 'Prediction (95% CL)')
        
        
      axes[ratio_n][i].plot(t_axis, (test_values-y_mean)/rmse, alpha = 1, label = 'Test values')

      #axes[ratio_n][i].grid()
      if i == 0:
        axes[ratio_n][i].set_ylabel('Data-Model (nT)')
        #axes[ratio_n][i].set_xticks(np.arange(t_axis[0], t_axis[-50],500))
      else:
        axes[ratio_n][i].set_yticks([])
        #axes[ratio_n][i].set_xticks(np.arange(t_axis[0], t_axis[-1],500))
        #axes[ratio_n][i].yaxis.tick_right()
        #axes[ratio_n][i].yaxis.set_label_position("right")
      #axes[ratio_n][i].set_xlabel('Time (min)')
      y_scale_0 = max(max(abs((test_values-y_mean)/rmse)),max(abs((y_max-y_mean)/rmse)))*1.1 #whichever is bigger between orange band and blue line
      if i == 0:
        y_scale = y_scale_0
      elif i == 1 and y_scale_0 > y_scale:
        y_scale = y_scale_0
      axes[ratio_n][i].set_ylim(-y_scale,y_scale)
      axes[ratio_n][i].set_xlim(5*a[i][st], 5*b[i][st])
      axes[ratio_n][i].set_xlabel('time (min)')

      handles, labels = axes[ratio_n][i].get_legend_handles_labels()
      # sort both labels and handles by labels
      labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
      #axes[ratio_n][i].legend(handles, labels, fontsize = 11)
      #axes[ratio_n][i].grid(True)

pic_name = 'bsdo_storm{}.pdf'.format(st)
print(peak_prefix+pic_name)

plt.savefig(base_dir+peak_prefix+pic_name, format = 'pdf')
plt.clf()

################# RMSE and R2 and comparisons #################

#~~~~~~~~~~~~~~~~~ RMSE and R2 computations for bootstrap and dropout ~~~~~~~~~~~~~~~~~#
mse = [{} for st in range(17)]
r2 = [{} for st in range(17)]
for st in range(17):
  mse[st]['bs'] = np.zeros(n_syst[st])
  mse[st]['do'] = np.zeros(200)
  r2[st]['bs'] = np.zeros(n_syst[st])
  r2[st]['do'] = np.zeros(200)

plt.figure(figsize=(6.4*6,4.8*3))
for st in range(17):
  plt.subplot(3,6,st+1)
  for j,i in enumerate(srt):
    n_max = int(max(DF[st][i]['n'])) + 1
    for n in range(n_max):
      df = DF[st][i][DF[st][i]['n'] == n]
      tf = np.array(TF[st][i]['sym'])
      vf = np.array(df['sym'])

      a1,a2 = a[j][st],b[j][st]
      if len(tf) == 0 or len(vf) == 0:
        print(st, a1, a2, vf,n)
      if n == 0:
        print(i,n_syst[st],len(tf),len(vf),st,n_max)

      mse_i = mean_squared_error(vf[a1:a2], tf[a1:a2])
      r2_i = r2_score(vf[a1:a2], tf[a1:a2])
      mse[st][i][n] = mse_i
      r2[st][i][n] = r2_i

    if j == 0:
      zz = 0.5
    plt.hist(np.sqrt(mse[st][i])*sym_std,label=i, density = True, alpha = zz)
    plt.title('storm {}'.format(st))
    plt.legend()
    plt.xlabel('RMSE (nT)')
    plt.tight_layout()

pic_name = 'MSE_bsdo_allstorms.pdf'
plt.savefig(peak_prefix+pic_name, format = 'pdf')

#~~~~~~~~~~~~~~~~~ Only bootstrap ~~~~~~~~~~~~~~~~~#
plt.figure(figsize=(6.4*6,4.8*3))
for st in range(17):
  plt.subplot(3,6,st+1)
  for i,j in enumerate([srt[0]]):
    if i == 0:
      zz = 0.5
    plt.hist(np.sqrt(mse[st][j])*sym_std,label=j, density = True, alpha = zz)
    plt.title('storm {}'.format(st))
    plt.legend()
    plt.xlabel('RMSE (nT)')
    plt.xlim(0)
    plt.tight_layout()

pic_name = 'MSE_bsdo_allstorms_Bootstrap.pdf'
plt.savefig(peak_prefix+pic_name, format = 'pdf')

# A function for rounding down with significant values considerations
def approxSignificant(x):
  '''x: array of values, each value is a touple of the mean and uncertainty'''
  count = np.ones(len(x))
  for j,i in enumerate(x):
    count[j] = 0
    if i[1] < 1:
      res = True
      i1 = i[1]
      while res:
        count[j] -= 1
        if 0.2 <= i1 < 1:
          res = False
        elif 0.1 < i1 < 0.2:
          count[j] -= 1
          res = False
        else:
          i1 = i1*10.0
    else:
      res = True
      i1 = i[1]/10.0
      while res:
        if 0.2 <= i1 < 1:
          res = False
        elif 0.1 < i1 < 0.2:
          count[j] -= 1
          res = False
        else:
          count[j] += 1
          i1 = i1/10.0
  a = 10**count
  #print(a,count)
  return(np.array([[int(np.round(x[z][0]/a[z]))*a[z], int(np.round(x[z][1]/a[z]))*a[z]] for z in range(len(x))]))

# Returning to RMSE and R2 bootstrap values
rmse = np.zeros((17,2))
r2_f = np.zeros((17,2))

for i in range(17):
  rmse[i][0] = np.mean(np.sqrt(mse[i]['bs'])*sym_std)
  rmse[i][1] = np.std(np.sqrt(mse[i]['bs'])*sym_std)
  r2_f[i][0] = np.mean(r2[i]['bs'])
  r2_f[i][1] = np.std(r2[i]['bs'])

rmse = approxSignificant(rmse)
r2_f = approxSignificant(r2_f)

# Values from other references
df = pd.DataFrame()

df['Set'] = np.array(['T{}'.format(int(i+1)) for i in range(17)])

#Siciliano
df["RMSE_LSTM_S"] = np.array([6.7, 8.9, 5.4, 7.2, 5.6, 10.7, 8.3, 16.3, 11.3, 8.5, 8.7, 17.5, 4.2, 5.6, 5.5, 9.0, 5.9])
df["R2_LSTM_S"] = np.array([0.89, 0.94, 0.95, 0.93, 0.95, 0.96, 0.95, 0.96, 0.75, 0.90, 0.89, 0.96, 0.94, 0.96, 0.95, 0.96, 0.97])
#Collado-Villaverde
df["RMSE_LSTM_S2"] = np.array([6.630, 8.913, 5.858, 6.683, 5.200, 8.584, 7.259, 13.340, 10.034, 7.693, 9.525, 15.184, 4.080, 6.431, 4.673, 7.882, 5.669])
df["R2_LSTM_S2"] = np.array([0.870, 0.939, 0.936, 0.922, 0.946, 0.971, 0.953, 0.965, 0.798, 0.907, 0.864, 0.966, 0.939, 0.932, 0.966, 0.969, 0.968])
#Iong
df["RMSE_LSTM_S3"] = np.array([5.863, 7.729, 4.281, 5.833, 4.927, 8.277, 6.841, 14.492, 10.190, 7.154, 8.512, 14.548, 3.886, 5.901, 4.976, 7.558, 5.030])
df["RMSE_CNN_S"] = np.array([7.2, 10.5, 5.6, 7.7, 6.5, 9.6, 8.2, 19.1, 12.4, 8.8, 10.5, 17.3, 4.6, 6.8, 5.9, 9.4, 6.3])

#~~~~~~~~~~~~~~~~~ Final RMSE and R2 comparison plots ~~~~~~~~~~~~~~~~~#
plt.figure(figsize=(14, 8), dpi=80)
for j,i in enumerate([rmse,r2_f]):
  quant = ['RMSE','R2']
  plt.subplot(1,2,j+1)
  #plt.title(quant[j])
  #plt.errorbar(x = df["RMSE_{}".format(i)], y = df["Set"], yerr = None, xerr = 1.96*df['RMSE_E_{}'.format(i)], fmt = 'o', capsize=5, label = "This work's results")
  plt.errorbar(x = i[:,0], y = df["Set"], yerr = None, xerr = 1.96*i[:,1], fmt = 'o', capsize=5, label = "This work", alpha = 0.85)
  plt.scatter(df['{}_LSTM_S'.format(quant[j])], df['Set'], color = 'orange', label = 'Siciliano et al. results')
  plt.scatter(df['{}_LSTM_S2'.format(quant[j])], df['Set'], color = 'red', marker = 'P', label = 'Collado-Villaverde et al. results')
  if j == 0:
    plt.scatter(df['{}_LSTM_S3'.format(quant[j])], df['Set'], color = 'green', marker = '*', label = 'Iong et al. results')

  plt.legend(fontsize = 12)
  plt.grid(axis = 'y')
  plt.yticks(ticks = df['Set'])
  plt.ylabel("Storm")
  plt.xlabel(["RMSE (nT)","R²"][j])
  plt.show
plt.savefig("CompareRMSE_Siciliano_200Runs.pdf", format = 'pdf')
