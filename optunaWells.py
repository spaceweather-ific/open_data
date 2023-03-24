
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import seaborn as sns
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.utils import resample
import pandas as pd

"""^[[32m[I 2022-04-24 16:37:15,418]^[[0m Trial 0 finished with value: 0.10394831169963993 and parameters: {'n_layers': 3, 'n_unit': 46, 'lr': 0.009928118071032398, 'lb': 90}. Best is trial 0 with value: 0.10394831169963993.^[[0m
"""

dnntype = "LSTM"
trials = "84"
SRT = "Bare" #Dropout or Bare
d_p = 0
if SRT == "Dropout":
  d_p = 1
with open("outfiles/outfile.500348.0.OptunaLSTM100NewDATA_Bare.out") as f:
  lines = f.readlines()
  mse = []
  mse_std = []
  n_layers = []
  n_unit = []
  lr = []
  lb = []
  #dropout_p = []
  #print(lines)  
  for l in lines:
#      print(l)
      try:
          a = l.split(' ')[2] == '+-'
          b = False
      except:
          b = True
          a = False
#      print(b)

      if not b and len(l.split("finished with value"))>1:
#          print(a,l[0])
          line_data = l.split("}")
          line_data = line_data[0].split(': ')
          
#          print(line_data)
          mse.append(float(line_data[1].split(' ')[0]))
          n_layers.append(int(line_data[3].split(',')[0]))
          #dropout_p.append(float(line_data[4].split(',')[0]))
          n_unit.append(int(line_data[4+d_p].split(',')[0]))
          lr.append(float(line_data[5+d_p].split(',')[0]))
          lb.append(int(line_data[6+d_p].split(',')[0]))
          best_index = int(l.split('Best is trial ')[1].split(' ')[0])

      elif a:
          line_data = l.split(' ')
#          print(a,line_data)
          mse_std_l = float(line_data[-1][:-1])
          #print(mse_std_l)
          mse_std.append(mse_std_l)

print(best_index)

the_bestest = []

for i in range(len(mse)):
    if np.abs(mse[i]-mse[best_index]) <= 1.96*mse_std[best_index]:
        bestest_i = [mse[i],mse_std[i],n_layers[i],n_unit[i],lr[i],lb[i]]#,dropout_p[i]]
        print(bestest_i)
        the_bestest.append(bestest_i)#,'best mse'])
#    else: # save these and the last variable is a hue
#        the_bestest.append([mse[i],mse_std[i],n_layers[i],n_unit[i],lr[i],lb[i],'not best mse'])

#print(len(the_bestest))
#print(the_bestest)

the_bestest = np.array(the_bestest)

norm = False

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 9), dpi=80, facecolor="w", edgecolor="k")

#for the lr log scale
hist, bins = np.histogram(lr, bins=10)
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

print(min(lr),max(lr))
print(hist, bins, logbins)

#histograms of all trials
axes[0,0].hist(n_layers,label = 'n_layers trials', density = norm, alpha = 0.70)
axes[0,1].hist(n_unit,label = 'n_unit trials', density = norm, alpha = 0.7)
axes[1,0].set_xscale('log')
axes[1,0].hist(lr,bins = logbins, label = 'lr trials', density = norm, alpha = 0.7)
axes[1,1].hist(lb,label = 'lookback trials', density = norm, alpha = 0.7)
#axes[2,0].hist(dropout_p,label = 'dropout_p', density = norm, alpha = 0.7)

#histograms of best trials
axes[0,0].hist(the_bestest[:,2], bins = np.histogram(n_layers)[1], label = 'best n_layers trials', density = norm)
axes[0,1].hist(the_bestest[:,3], bins = np.histogram(n_unit)[1], label = 'best n_unit trials', density = norm)
axes[1,0].set_xscale('log')
axes[1,0].hist(the_bestest[:,4], bins = logbins, label = 'best lr trials', density = norm)
axes[1,1].hist(the_bestest[:,5], bins = np.histogram(lb)[1], label = 'best lookback trials', density = norm)
#axes[2,0].hist(the_bestest[:,6], bins = np.histogram(dropout_p)[1], label = 'best dropout_p', density = norm)

axes[0,0].legend()
axes[0,1].legend()
axes[1,0].legend()
axes[1,1].legend()
#axes[2,0].legend()

#fig.title('Frequency of hyperparameters in 500 optuna trials')
plt.savefig('also_best_optuna'+dnntype+'_'+SRT+'.png')

the_bestest_df = pd.DataFrame(the_bestest, columns = ['mse_mean','mse_std','n_layers','n_unit','lr','lb'])#,'dropout_p'])#,'best_or_nah'])
print(the_bestest_df)

plt.figure(figsize = (12,12))
g = sns.PairGrid(the_bestest_df, corner = True, vars = ['n_layers','n_unit','lr','lb'])#,'dropout_p'])#, hue = 'best_or_nah')
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
g.savefig('optuna'+dnntype+'Bestest_scatterplots_'+SRT+'.png')
