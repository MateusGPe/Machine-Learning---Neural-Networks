import csv
import pandas as pd
import numpy as np
import sys

def buildData(df,col,colNum):
	uniqueVals = col.unique()
	newcol = col
	attrValues = {}
	j = 0
	for val in uniqueVals:
		attrValues[val] = j
		j = j+1
	new = []
	for index,row in col.iteritems():
		new.insert(index,attrValues[row])
	df[colNum] = new
	newcol = (df[colNum] - df[colNum].mean())/df[colNum].std()
	df[colNum] = newcol
	return

def lastCol(df, col,colNum):
	if col.dtype == np.int64 or col.dtype == np.float64:
		#newcol = (col - col.min())/(col.max()-col.min())
		newcol = (col >= col.mean())*1
		df[colNum] = newcol
	else:
		uniqueVals = col.unique()
		newcol = col
		attrValues = {}
		j = 0
		for val in uniqueVals:
			attrValues[val] = j
			j = j+1
		new = []
		for index,row in col.iteritems():
			new.insert(index,attrValues[row])
		df[colNum] = new
		newcol = (df[colNum] - df[colNum].min())/(df[colNum].max()-df[colNum].min())
		df[colNum] = newcol
	return

inputFile = sys.argv[1]
outputFile = sys.argv[2]

data = pd.read_table(inputFile,sep='\t|,|:|\s+',index_col = False,header=None, engine = 'python')
#data = pd.read_table(inputFile,sep=",",index_col = False,header=None)
#data = pd.read_table('G:\MachineLearning\Assignments\Assignment3\Datasets\Iris\iris.data.txt', sep=",",index_col=False,header=None)

#print data.dtypes
newDf = pd.DataFrame()
data.replace(to_replace="[?]",value=np.nan,regex=True,inplace=True)
data = data.dropna()

for col in range(0,len(data.columns)-2):
	if data[col].dtype == np.int64 or data[col].dtype == np.float64:
		newcol = ((data[col]-data[col].mean())/data[col].std())
		newDf[col]= newcol
	else:
		#print data[col].unique()
		buildData(newDf,data[col],col)
#iterate - a particular column(Series in pandas) in the DF 
#for index,row in data[len(data.columns)-1].iteritems():
#	print row
#iterate - entire DF
#for index,row in data.itertuples():
#	print row
#out_col = pd.Series()

lastCol(newDf,data[len(data.columns)-1],len(data.columns)-1)

#print newDf.shape
newDf.to_csv(outputFile,sep=',',index = False)
#newDf.to_csv("adultOutput.csv",sep=',',index=False)
#newDf.to_csv("G:\MachineLearning\Assignments\Assignment3\Datasets\Iris\iris.data.csv",sep=',',index=False)
