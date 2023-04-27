# ACE_data directory

* The **ACE_data** folder contains the output of the preprocessing of the raw data obtained from [OMNIWeb](https://omniweb.gsfc.nasa.gov).
	* The file ```all_data.gzip``` has every item, including the timestamps of each value.
	* The other ```.gzip``` files contain only the feature variables and the corresponding split of the total data.
	* The ```.json files``` contain info about the mean and standard deviation of each feature variable, the number of values corresponding to a different storm, the names of the feature variables, and the name of the time key.
