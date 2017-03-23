The programs require Python2 to be installed, and the following Python libraries: csv, pandas and numpy.

How to execute:
For the data pre-processing, run dataClean.py with the following two parameters:
1. input file path
2. output file path

Example: 

python dataClean.py housing.data housingCleaned.csv

To train the neural network, and see the results, execute neuralNets.py, with the following parameters:
1. Cleaned data filepath
2. Percentage of data to be used for training
3. Error tolerance rate
4. Number of Hidden layers
5. Number of units in each hidden layer, separated by a space

Example:
python neuralNets.py housingCleaned.csv 80 0.05 2 2 3

