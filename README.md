This repository contains the results of the project related to the paper "Forecasting geomagnetic storm disturbances and their uncertainties using deep learning".

##Contents
* "ACE_data" folder contains the output of the preprocessing of the raw data obtained from OMNIWeb (https://omniweb.gsfc.nasa.gov).
	* The file "all_data.gzip" has every item, including the timestamps of each value.
	* The other .gzip files contain only the feature variables and the corresponding split of the total data.
	* The .json files contain info about the mean and standard deviation of each feature variabe, the number of values corresponding to a different storm, the names of the feature variables, and the name of the time key.

* "BS_Models" folder contains all of the 200 models consctructed for the bootstrap uncertainty estimation approach, in the form of .h5 files.
* "DO_Models" folder contains the model constructed for the dropout uncertainty estimation approach, in the form of a .h5 file, also.
* "TestStormPredictions.pdf" is a pdf file containing high-definition graphs of each of the test storm and the prediction aggregates from both bootstrap and dropout estimations.
* "PeaksOfTestStormPredictions.pdf" is the same as the previous file but only showing the section of the test storms with the most dramatic activity.
