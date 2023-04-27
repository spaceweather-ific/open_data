# Open data

This repository contains software and results of the project related to the paper entitled _**Forecasting geomagnetic storm disturbances and their uncertainties using deep learning**_.

## Contents

The content of the repository is the follwong:

* The **ACE_data** folder contains the output of the preprocessing of the raw data obtained from [OMNIWeb](https://omniweb.gsfc.nasa.gov).
	* The file ```all_data.gzip``` has every item, including the timestamps of each value.
	* The other ```.gzip``` files contain only the feature variables and the corresponding split of the total data.
	* The ```.json files``` contain info about the mean and standard deviation of each feature variable, the number of values corresponding to a different storm, the names of the feature variables, and the name of the time key.
	
* The ***BO_Models*** folder contains all of the 200 models consctructed for the block-bootstrap uncertainty estimation approach, in the form of ```.h5``` files.

* The ***DO_Models*** folder contains the model constructed for the dropout uncertainty estimation approach, also in the form of a ```.h5``` file.

* The ***Plots*** folder contains:
	* The ```PeaksOfTestStormPredictions.pdf``` is the same as the previous file but only showing the section of the test storms with the most dramatic activity (i.e. peaks).
	* The ```TestStormPredictions.pdf``` is a pdf file containing high-definition graphs of each of the final test storm and the prediction aggregates from both bloack-bootstrap and dropout estimations.
	* The ```MultiHour_StormPredictions.pdf```is a pdf file containing high-definition graphs of multi-hour ahead predictions of each test storm.

* The **config** folder just contains the ```config.ini``` file use by the ```train.py``` script to set other parameters like [Optuna](https://optuna.org/) intervals, hyper-parameter values, data used, and number of runs or [Optuna](https://optuna.org/) trials.
	
* The **lib** folder contains:
	* The ```trainUtils.py``` script with functions like the one that creates the datasets to be used in the sequential models, the one that does so for each particular storm, and the cyclical learning rate custom function.
	* The ```trainModels.py``` script with the concrete dropout class, the custom ```MonteCarloLSTM``` class and the functions that build models with the 4 proposed architectures, and the objective function for the optuna runs.
	
* The ***predictionsAndTest_Values*** folder includes the results of both the block-bootstrap and dropout models, and also the _testValues_ to be used in the script ```SpaceWeatherPrediction_Analysis.py```.

* The ```LICENSE``` is the licence file.
	
* The ```SpaceWeatherPrediction_Analysis.py``` produces the prediction graphs shown in the paper, and also the comparison plot between the RMSE values of all the test storms and the same values from other references.
	
* The ```optunaWells.py``` script takes the output of the [Optuna](https://optuna.org/) trials to calculate the confidence interval of the MSE of the best [Optuna](https://optuna.org/) trial. Then it collects all trials which gave an MSE within the calculated confidence interval. Finally, it produces graphs showing the pair-plot scatter distribution of the collected best trials for the hyper-parameters, as well as their histograms. These graphs show where the best values of hyper-parameters are collected, roughly.

* The ```pre_process.py``` is the script that takes the data from [OMNIWeb](https://omniweb.gsfc.nasa.gov) and produces the files contained in the **ACE_data** folder.

* The ```requirements.txt``` file includes the required packages to be installed to run the libraries in these scripts. We run them under [NVIDIA GPU CUDA Version: 12.0  Driver Version: 525.85.12](https://www.nvidia.com/Download/driverResults.aspx/198879/es/).

* The ```train.py``` is the script used to build the models.
	1. first this script is used with the [Optuna](https://optuna.org/) argument ```--optuna``` to obtain the optimised hyper-parameters; the output is saved in a text file later used in the script ```optunaWells.py``` to obtain the graphs shown in the paper.
	2. Afterwards, with the chosen hyper-parameters, [Optuna](https://optuna.org/) was left untriggered and, instead, the block-bootstrap or dropout models are built and predictions are saved in ```.csv``` files.
	3. Finally, feature importance is activated and its corresponding graph is produced.


	









