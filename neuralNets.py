import csv
import pandas as pd
import numpy as np
import sys

def sigmoid(x):
	return (1/(1+np.exp(-x)))

def createWghtArray(n,m):
	w = np.random.ranf((n,m))*2*INIT_EPSILON-INIT_EPSILON
	#print w
	return w

def readData(path):
	return np.genfromtxt(path,delimiter=',',skip_header = 1)

def addBiasUnit(arr):
	r = np.ones((arr.shape[0],1))
	return np.concatenate((r,arr),axis=1) #axis=1 --> column

def split(arr):
	numrows = arr.shape[0]
	numcols = arr.shape[1]
	num = np.rint(numrows*int(sys.argv[2])/100.0).astype(int)
	np.random.shuffle(arr)
	trian,test,train_out,test_out = arr[0:num,0:numcols-1],arr[num:,0:numcols-1],arr[0:num,numcols-1:],arr[num:,numcols-1:]
	return trian,train_out,test,test_out

def msd(output,actual):
	sqrd = (actual-output)**2
	msdErr = np.sum(sqrd)/len(sqrd)
	return msdErr

def forwardBackwardProp(lr,w,layers,train_data,train_output):
	a={}	
	epsilon = {}
	deltaW = {}
	for t in range(0,len(train_data)-1):
		# Forward Propagation
		a[0] = train_data[t]
		for i in range(1,len(layers)):
			a[i-1] = np.append(1,a[i-1])
			a[i] = sigmoid(np.dot(a[i-1],w[i-1]))
		output = a[len(layers)-1]
		
		#Back propagation
		epsilon[len(layers)-1] = (train_output[t]-output)  * output * (1-output)	
		for h in range(len(layers)-2,0,-1):
			#except for input and output layers - this condition to take care of the bias units
			if h+1 == len(layers)-1:
				epsilon[h] = a[h]*(1-a[h])*np.dot(epsilon[h+1],w[h].transpose())
				deltaW[h] = lr * np.dot(a[h][:,None],epsilon[h+1][None,:])
			else:
				epsilon[h] = a[h]*(1-a[h])*np.dot(epsilon[h+1][1:],w[h].transpose())
				deltaW[h] = lr * np.dot(a[h][:,None],epsilon[h+1][None,:][:,1:])
		deltaW[0]=lr*np.dot(a[0][:,None],epsilon[1][None,:][:,1:])
		for i in range(0,len(layers)-1):
			w[i] = w[i] + deltaW[i]
	return

def predict(w,train_data):
	a ={}
	output = {}
	a[0] = train_data
	for i in range(1,len(layers)):
		a[i-1] = addBiasUnit(a[i-1])
		a[i] = sigmoid(np.dot(a[i-1],w[i-1]))
	output = a[len(layers)-1]
	return output

data = readData(sys.argv[1])
train_data,train_output,test_data,test_output = split(data)
numCols = data.shape[1]-1 #excluding output column
numRows = data.shape[0]
numClasses = np.unique(train_output).size

INIT_EPSILON = 0.01
layers = [numCols]
lr = 0.1
errorTolerence = float(sys.argv[3])
hiddenLayersNum = int(sys.argv[4])
for i in range(0, hiddenLayersNum):
	layers.append(int(sys.argv[5+i]))
layers.append(1)

w={}
for i in range(0,len(layers)-1):
	w[i] = createWghtArray(layers[i]+1,layers[i+1])
e = 0
numOfIterations = 50
for itr in range(0,numOfIterations):
	forwardBackwardProp(lr,w,layers,train_data,train_output)
	output = predict(w,train_data)
	err = msd(output,train_output)
	if err < errorTolerence:
		print output
		break
	e = err
	#print err
	#if itr == numOfIterations -1:
		#print output


out = predict(w,test_data)
err = msd(out,test_output)



for i in range(0,len(layers)-2):
	print "Hidden Layer "+ str(i)+": "
	for j in range(0,layers[i]+1):
		print "\tNeuron"+ str(j)+ " weight - " + str(w[i][j])
print "Output Layer: "
for j in range(0, layers[len(layers) - 2]+1):
	print "\tNeuron"+ str(j)+ " weight - " + str(w[len(layers) - 2][j])

print "Total Training error: "+ str(e)
print "Total Test error: "+ str(err)