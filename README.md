This repository contains software and results of the project related to the paper "Forecasting geomagnetic storm disturbances and their uncertainties using deep learning".

##Contents
* "ACE_data" folder contains the output of the preprocessing of the raw data obtained from OMNIWeb (https://omniweb.gsfc.nasa.gov).
	* The file "all_data.gzip" has every item, including the timestamps of each value.
	* The other .gzip files contain only the feature variables and the corresponding split of the total data.
	* The .json files contain info about the mean and standard deviation of each feature variabe, the number of values corresponding to a different storm, the names of the feature variables, and the name of the time key.

* "requirements.txt" includes the packages to be installed to run the libraries in these scripts.
* "pre_process.py" is the script that takes the data from omniweb and produces the files contained in the "ACE_data" folder.
* "train.py" is the script used to build the models. It is first used with the optuna argument --optuna to obtain the optimized hyperparameters; the output was saved in a text file later used in the script "optunaWells.py" to obtain the graphs shown in the paper. Second, with the chosen hyperparameters, optuna was left untriggered and, instead, the bootstrap or dropout models are built and predictions are saved in .csv files. Third, feature importance is activated and its corresponding graph is produced.
* "config/config.ini" is referenced to "train.py" to set other parameters like optuna intervals, hyperparameter values, data used, and number of runs or optuna trials.
* "lib" folder contains: a) The "trainUtils.py" script with functions like the one that creates the datasets to be used in the sequential models, the one that does so for each particular storm, and the cyclical learning rate custom function, and b) the "trainModels.py" script with the concrete dropout class, the functions that build models with the 4 proposed architectures, and the objective function for the optuna runs.
* "optunaWells.py" script takes the output of the optuna trials to calculate the confidence interval of the MSE of the best optuna trial. Then it collects
all trials which gave an MSE within the calculated confidence interval. Finally, it produces graphs showing the pairplot scatter distribution of the collected best trials
for the hyperparameters, as well as their histograms. These graphs show where the best values of hyperparameters are collected, roughly.
* "BS_Models" folder contains all of the 200 models consctructed for the bootstrap uncertainty estimation approach, in the form of .h5 files.
* "DO_Models" folder contains the model constructed for the dropout uncertainty estimation approach, in the form of a .h5 file, also.
* "predictionsAndTest_Values" folder includes the results of both the bootstrap and dropout models, and also the testValues for use in the script "SpaceWeatherPrediction_Analysis.py"
* SpaceWeatherPrediction_Analysis.py" produces the prediction graphs shown in the paper, and also the comparison plot between the RMSE values of all the test storms and the same values from other references.
* "TestStormPredictions.pdf" is a pdf file containing high-definition graphs of each of the final test storm and the prediction aggregates from both bootstrap and dropout estimations.
* "PeaksOfTestStormPredictions.pdf" is the same as the previous file but only showing the section of the test storms with the most dramatic activity.
