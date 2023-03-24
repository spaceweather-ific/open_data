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
from sklearn.metrics import r2_score, mean_squared_error

import derivative
from derivative import dxdt
from tqdm.notebook import tqdm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, ConvLSTM2D, Activation, AveragePooling2D, MaxPooling2D, Flatten, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import models

from pandas.plotting import table

from attention import Attention

import matplotlib
import optuna
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import sys

# First, concrete dropout wrapper taken from https://github.com/yaringal/ConcreteDropout/blob/master/spatial-concrete-dropout-keras.ipynb

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import InputSpec, Dense, Wrapper, Input, concatenate
from tensorflow.keras.models import Model


def finite_diff(x,k,plot_d = False,aa = 0,bb = 0):
  """This function produces an array with equal shape as x,
  each value in this array is the difference between the average of k numbers
  in x before the index and the k numbers in x after the index of the value
  x: an N sized 1D array
  k: width of computation
  plot: to plot or not
  a: start of storm to plot
  b: end of storm to plot"""
  dx = []
  for i in range(len(x)):
    a = max(i-k,0)
    b = min(i+k,len(x))
    if a == 0 or b == len(x):
      dx.append(0)
    else:
      a_m = np.mean(x[i-k:i])/k
      b_m = np.mean(x[i:i+k])/k
      dx.append(b_m-a_m)
  print('hola')
  if plot_d:
    xc = x[aa:bb]
    dxc = dx[aa:bb]
    t = np.array(range(len(xc)))*5
    plt.figure(figsize=(25,12))
    ax = plt.subplot(211)

    ax.plot(t,np.array(dxc), label='derivative {}'.format(k))

    ax.grid()
    ax.xaxis.set_major_locator(MultipleLocator(500))
#ax.xaxis.set_minor_locator(AutoMinorLocator(50))
    ax.legend()

    ax = plt.subplot(212)
    ax.plot(t,xc, label = 'x')

    ax.xaxis.set_major_locator(MultipleLocator(500))

    ax.grid()
    ax.legend()
    plt.savefig('sample_derivative_calc.png')
  return(np.array(dx))

# dropout class for test taken from https://fairyonice.github.io/Measure-the-uncertainty-in-deep-learning-models-using-dropout.html
class KerasDropoutPrediction(object):
    def __init__(self,model):
        tf.compat.v1.disable_eager_execution()
        print(tf.keras.backend.learning_phase())
        self.f = tf.keras.backend.function(
                [model.layers[0].input, 
                 tf.keras.backend.learning_phase()],
                [model.layers[-1].output])
    def predict(self,x, n_iter=10):
        result = []
        for _ in range(n_iter):
            result.append(self.f([x , 1]))
        result = np.array(result).reshape(n_iter,len(x)).T
        return result

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def custom_lr_cycle(epoch, lr):
    """Function that implements one cycle lr rate as by https://arxiv.org/pdf/1506.01186.pdf
    It's centered around the best value given by optuna and cycles within the
    optuna interval with a stepsize of 10
    """
#    epoch=100
    log_lr_mean = np.log10(lr)
    log_lr_std = 0.5
    step_size = 10
    x = epoch % 10
    m = 4*log_lr_std/step_size
    #print("lr_stuff",x, epoch, m)
    if 2*x < step_size:
        log_lr = log_lr_mean-log_lr_std + m*x
        #print(log_lr_mean, log_lr_std, m, x, log_lr)
        return(10**log_lr)
    else:
        log_lr = log_lr_mean - log_lr_std + step_size*m - m*x
        return(10**log_lr)


# ---------------------------------------------------------------
# Read Configuration File
# ---------------------------------------------------------------
def readConfig(configFile, check=False) :

    parameters = {}

    try:
        InputFile = open(configFile, 'r')
        while 1:
            line = InputFile.readline()
            if not line: break
            if '#' not in line and len(line) > 1 :
                command = line.split("=")
                parameters[command[0].strip()] = command[1].strip()
        InputFile.close()

       
        return parameters
    except IOError:
        errmsg = " File %s does not exist!" % configFile
        print( bcolors.FAIL + errmsg + bcolors.ENDC)
        sys.exit()




# Data Normalization to make training easier
def normalize(data, data_mean = "None", data_std = "None"):
    #print("bye", data_mean == "None", data_std == "None")
    if type(data_mean) == str or type(data_std) == str: # Unless both values are specified, conditional checks out and mean and std are calculated
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
    return data_mean, data_std, (data - data_mean) / data_std

"""# Helper Functions"""


# Create a dataframe for a given model
def getTableSys(list_models, dtype) :
  df = None

  if not len(list_models) : return None, None

  # load configuration of the dnn
  import json
  file_name =os.path.join(output_dir, dnn_name[dnn_type]+'.json')
  with open(file_name) as json_file:
      settings = json.load(json_file)

  # create a sample model that will be overwritten
  x_train, y_train = create_dataset(data['Train'], settings['lb'], settings['lf'], len(selected_features),interface=[],step=settings['step'])
  if dnn_type == 3 :
    model = build_dnn(dnn_type,x_train.shape[1:], y_train.shape[1],  lr=learning_rate, nlayers=settings['nlayers'], nD=settings['nD'])
  else :
    model = keras.models.load_model(list_models[0])


  i = 0
  for model_ in list_models:
#      print(model_)
      if not dnn_type == 3 :
        model = keras.models.load_model(model_)
      else :
        model.load_weights(model_)


      detail_test = createTestTable(model,settings,i)
      try :
          df = pd.merge(detail_test, df, on=['Dataset'])
      except :
          df = copy.deepcopy(detail_test)
      i += 1

  # Save minimum values for RMSE and R2
  minRes = df.columns[df.columns.str.startswith("RMSE")]
  df['minRMSE '+dtype] = df[minRes].min(axis=1).round(decimals=1)
  minRes = df.columns[df.columns.str.startswith("R2")]
  df['minR2 '+dtype] = df[minRes].min(axis=1).round(decimals=2)

  # Keep just the right columns
  resdf = df.filter(['Dataset','minRMSE '+dtype,'minR2 '+dtype], axis=1)

  return df, resdf

# Get One storm data
def getStorm(all_data,stat_data,selected_features,dnn_type,lb,lf,dset_key,dtype = 'Test', predict_time_derivative = False)  :
  test_df = all_data.loc[all_data['Type'] == dtype]
  date_time_key = stat_data[dtype]['time_key']
#  print('d_time_key', date_time_key, dset, stat_data[dtype]['breaks'])

  # Create a new dataframe normalized to use for prediction
  features = test_df.loc[test_df['Dataset'] == dset_key][selected_features]
  features.index = test_df.loc[test_df['Dataset'] == dset_key][date_time_key]
  train_means = np.array(all_data[all_data.Type == 'Train'].mean(axis = 0)[selected_features])
  train_stds = np.array(all_data[all_data.Type == 'Train'].std(axis=0)[selected_features])
  _, _, v_test = normalize(features.values, train_means, train_stds)
  v_data = pd.DataFrame(v_test,columns=selected_features)

  # Create the actual dataset and predict
  x_storm, y_storm = create_dataset(v_data, dnn_type,lb, lf, selected_features,len(selected_features), interface=stat_data[dtype]['breaks'],
                                    step=1, predict_time_derivative = predict_time_derivative)

  return x_storm, y_storm, train_means, train_stds



def save_df_as_image(df, path):

    import matplotlib
    import seaborn as sns
    # Set background to white
    norm = matplotlib.colors.Normalize(-1,1)
    colors = [[norm(-1.0), "white"],
            [norm( 1.0), "white"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    # Make plot
    plot = sns.heatmap(df, annot=True, cmap=cmap, cbar=False)
    fig = plot.get_figure()
    fig.savefig(path)

# Creates a table with RMSE and R2 for each of the Datasets (Storms)
def createTestTable(model,all_data,stat_data, dnn_type,lb, lf,selected_features,step,settings=None,i=0) :

  test_v_array = []
  # Select Test Data
  test_df = all_data.loc[all_data['Type'] == 'Test']

  # Run over each of the Storms
  for val in test_df.Dataset.unique() :


    date_time_key = stat_data['Test']['time_key']
    # Create a new dataframe normalized to use for prediction
    features = test_df.loc[test_df['Dataset'] == val][selected_features]
    features.index = test_df.loc[test_df['Dataset'] == val][date_time_key]
    v_mean, v_std, v_test = normalize(features.values)
    v_data = pd.DataFrame(v_test,columns=selected_features)

    # Create the actual dataset and predict
    if settings :
      x_v, y_v = create_dataset(v_data, settings['lb'], settings['lf'], len(selected_features),step=1)
    else :
      x_v, y_v = create_dataset(v_data, dnn_type, lb, lf, selected_features,len(selected_features), step=step)
    predict_v = model.predict(x_v)


    # Modifications to allow to work for each architecture
    if predict_v.ndim == 3 :
      predict_v = np.reshape(predict_v, (len(predict_v), 1))

    n_lf = 0
    if predict_v.shape[1] > 1 :
      n_lf = lf - 1

    # Compute RMSE and R2
    rmse =  np.sqrt( mean_squared_error (predict_v[:, n_lf]*v_std[0], y_v[:, n_lf]*v_std[0]) )
    r2 =  r2_score(predict_v[:, n_lf]*v_std[0], y_v[:, n_lf]*v_std[0])
    test_v_array.append([val, rmse, r2])


  columns =['Dataset', 'RMSE', 'R2']
  if i > 1 :
    columns =['Dataset', 'RMSE_'+str(i), 'R2_'+str(i)]

  # Return dataframe
  test_v_df = pd.DataFrame(test_v_array, columns=columns)
  #save_df_as_image(test_v_df, "table.png")
  return test_v_df




# Plot RMS for prediction and truth
def do_diff(ax, ypred, ytest, n) :
  from sklearn.metrics import r2_score, mean_squared_error

  #plt.figure(figsize=(10,10))

  rmse =  np.sqrt( mean_squared_error (ypred,ytest) )
  r2 =  r2_score(ytest,ypred)
    
  diffs=pd.melt(pd.DataFrame(ypred-ytest, columns=['SYMH']))
  #diffs=pd.melt(pd.DataFrame(ypred-ytest, columns=['BX nT (GSE GSM)']))
  sns.histplot(diffs, x='value', hue='variable', alpha=0.5, kde=True, element='step',legend=False, ax=ax)
  ax.set_xlabel('SYMH Predicted - SYMH Expected (nT)')
  #ax.set_title( 'RMSE = {:.2f} (nT) & R2 = {:.3f}'.format(rmse,r2) )
  ax.text(0.6, 0.85, 'RMSE (nT) = {:.2f}\nR2 = {:.3f}'.format(rmse,r2), fontsize=24, transform=ax.transAxes)
  ax.set_xlim(-50, 50)
  ax.set_facecolor("w")


# Plot predictions and truth (calls do_diff)
def plot_predictions(output_dir,ptype,prediction_plot,y_test,plotlen, std=1, mean=0) :

  if prediction_plot.ndim == 3 :
    prediction_plot = np.reshape(prediction_plot, (len(prediction_plot), 1))
  if prediction_plot.shape[1] > 1 :
    fig, axes = plt.subplots(nrows=int(lf/2), ncols=4, figsize=(30, 40), dpi=80, facecolor="w", edgecolor="k")
    for n in range(lf):
      plt.figure(facecolor="w")
      matplotlib.rc('axes',edgecolor='black')
      axes[n // 2, n % 2].plot(y_test[:plotlen, n]*std+mean, label='Test data', color="royalblue", linewidth=2)
      axes[n // 2, n % 2].plot(prediction_plot[ : plotlen , n]*std+mean, label='Prediction forward {} min'.format((n+1)*5), color="red", linewidth=2)
      axes[n // 2, n % 2].legend()
      do_diff(axes[n // 2, n % 2 + 2], prediction_plot[ : , n]*std, y_test[: , n]*std, n)
    plt.tight_layout()
    plt.savefig(output_dir+'Predictions_'+ptype+'.png')
    plt.close(fig)

  else :
    n = 0
    plt.figure(facecolor="w")
    matplotlib.rc('axes',edgecolor='black')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 10), dpi=80, facecolor="w")
    axes[0].set_facecolor("w")
    axes[0].plot(y_test[:plotlen, n]*std+mean, label='Test data', color="royalblue", linewidth=2)
    #print("Test data: ", y_test[:plotlen, n]*std+mean)
    axes[0].plot(prediction_plot[ : plotlen , n]*std+mean, label='Prediction', color="red", linewidth=2)
    #print("Prediction: ", prediction_plot[ : plotlen , n]*std+mean)
    axes[0].legend(loc="upper left",fontsize=18,title_fontsize=12,shadow=True, fancybox=True, bbox_to_anchor=(0.05,.15), facecolor="w")
    #axes[0].set_ylabel('Bx (nT)', fontsize=18)
    axes[0].set_ylabel('SYMH (nT)', fontsize=18)
    axes[0].set_xlabel('Time', fontsize=16)
    do_diff(axes[1], prediction_plot[:, n]*std, y_test[:, n]*std, n)
    plt.tight_layout()
    plt.savefig(output_dir+'Prediction_'+ptype+'.png')
    plt.close(fig)

    return fig

# Create a few plots with the values for all data
def show_raw_visualization(data):
    fig, axes = plt.subplots(
        nrows=int(len(data.columns)/2)+1, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(data.columns)):
        key = data.columns[i]
        t_data = data[key]
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            title="{}".format(key),
            rot=25,
        )
        ax.legend(key)
    plt.tight_layout()


# Create a few plots with the values for all data
def show_raw_histograms(data):
    fig, axes = plt.subplots(
        nrows=int(len(data.columns)/2)+1, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(data.columns)):
        key = data.columns[i]
        t_data = data[key]
        t_data.head()
        ax = t_data.hist(
            ax=axes[i // 2, i % 2],
            bins = 100
        )
        ax.legend(key)
    plt.tight_layout()


"""# Configuration and dataset creation"""

# convert an array of values into a dataset matrix
def create_dataset(dataset, dnn_type, lb, lf, selected_features, features, predict=0, interface=[], step=1, predict_time_derivative = False):
    all_points=False
    dataX, dataY = [], []
    n_lf = lf
    # Select correct column to predict
    predict_column = selected_features[predict]
    #if predict_time_derivative:
    #    predict_column = 'SYM/H nT' # There's a new extra time derivative column of SYMM/H
    #    features += -1

   
    for i in range(0,len(dataset) - lb - lf, step):

        removeData = False
        for x in interface :
          if x <= i + lb + lf - 1 and x >= i :
              removeData = True

        if removeData :
            continue

        a = dataset.loc[i: i + lb - 1, dataset.columns != 'SYM/H nT/min'].values
        dataX.append(a)
        if not all_points :
          dataY.append(dataset.loc[i + lb + lf - 1, predict_column])
          n_lf = 1
        else :
          dataY.append(dataset.loc[i + lb : i + lb + lf - 1 , predict_column])

    

    if dnn_type == 0 :
      return np.reshape(dataX, (len(dataX), lb, 1, features)), np.reshape(dataY, (len(dataY), n_lf))
    if dnn_type == 1 or dnn_type == 3 :
      return np.reshape(dataX, (len(dataX), lb, features)), np.reshape(dataY, (len(dataY), n_lf))
    if dnn_type == 2 :
      return np.reshape(dataX, (len(dataX), lb, 1, features)), np.reshape(dataY, (len(dataY), 1))


"""# Checking Performance"""

def histplot(history, metrics='mse'):
    hist = pd.DataFrame(history.history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    hist.plot(y=['loss', 'val_loss'], ax=ax1)
    min_loss = hist['val_loss'].min()
    ax1.hlines(min_loss, 0, len(hist), linestyle='dotted',
               label='min(val_loss) = {:.3f}'.format(min_loss))
    ax1.legend()
    hist.plot(y=[metrics, 'val_{}'.format(metrics)], ax=ax2)
    min_metrics = hist['val_{}'.format(metrics)].min()
    ax2.hlines(min_metrics, 0, len(hist), linestyle='dotted',
               label='min(val_{}) = {:.3f}'.format(metrics, min_metrics))

    fig.savefig("val_loss.png")

def CheckTestStorms(output_dir,all_data, stat_data, selected_features,dnn_type,dnn_name,lb, lf, model):
    """# Test Each Storm

    Show Data vs Prediction for each Test Storm separately (all points predicted, step is 1)
    """
    

    plotlen = 100000
    keys = all_data.loc[all_data['Type'] == 'Test']['Dataset'].unique()

    plt.close('all')

    for dset_key in keys :
      x_storm, y_storm, storm_mean, storm_std = getStorm(all_data,stat_data,selected_features,dnn_type,lb, lf,dset_key)
      prediction_plot = model.predict(x_storm)

      #display(HTML('<h2>Storm Test '+dset_key+'</h2>'))
      #print("Print input from plotter: ", prediction_plot,y_storm,plotlen,storm_std[0],storm_mean[0])

      nplt = plot_predictions(output_dir,"all_norm_"+str(dnn_name)+"_"+str(dset_key),prediction_plot,y_storm,plotlen,storm_std[0],storm_mean[0])
      #display(nplt)
      plt.close('all')

      nplt = plot_predictions(output_dir,"all_"+str(dnn_name)+"_"+str(dset_key),prediction_plot,y_storm,plotlen,1,0)
      #display(nplt)
      plt.close('all')

    # Create a table similar to Sicilianos paper with values for each storm
    detail_test = createTestTable(model,all_data,stat_data,dnn_type,lb, lf,selected_features, step=1)
    #print(detail_test)

    # set fig size
    fig, ax = plt.subplots()
    # no axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # no frame
    ax.set_frame_on(False)
    # plot table
    tab = table(ax, detail_test, loc='upper right')
    # set font manually
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)

    plt.savefig(output_dir+"/"+str(dnn_name)+'_withoutSys.png')




def StormTable(data,name):
    # set fig size
    fig, ax = plt.subplots()
    # no axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # no frame
    ax.set_frame_on(False)
    # plot table
    tab = table(ax, data, loc='upper right')
    # set font manually
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    # save the result
    plt.savefig(name)
    plt.close()
