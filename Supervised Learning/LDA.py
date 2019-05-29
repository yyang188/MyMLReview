import numpy as np
import pandas as pd
import math
from sklearn import datasets
from sklearn.metrics import confusion_matrix


# Randomly_split dataset

def random_split(data,portion = 0.8):
	m = data.shape[0]
	n = data.shape[1]
	idx =  np.random.choice(m,m,replace = False)  
	data_train = data.iloc[idx[0:np.int(round(m*portion))],:]
	data_test = data.iloc[idx[np.int(round(m*portion)):],:]

	return data_train,data_test

# Calculate Pooled covariance:

# data: a data frame
# lebel: a string indicating label column
# m: # of rows of dataset
# n: # of columns of dataset
# num_c: # of classes

def Pool_cov(data,label,m,n,num_c):

	Mu_k = np.array(data.groupby(label).mean())
	c = np.array(data[label])
	within_diff = np.array(data)
	

	for i in range(m):
		within_diff[i,0:n-1] = within_diff[i,0:n-1]-Mu_k[c[i],:]

	
	sigma_hat = np.zeros([n-1,n-1])

	for i in range(num_c):
		each_c = within_diff[c == i,:]
		each_c = each_c[:,0:n-1]
		sigma_hat += each_c.T @ each_c

	sigma_hat = sigma_hat*(1/(m-num_c))
	
	return sigma_hat

# input:
# Data: a pandas data frame with label
# label: the column in dataset which should be used as label 

def LDA_classifier(data,label):

	m = data.shape[0]
	n = data.shape[1]
	num_c = len(set(data[label]))

	######## Calculate Pooled covariance ##########

	sigma_hat  = Pool_cov(data,label,m,n,num_c)
	sigma_hat_inv = np.linalg.inv(sigma_hat)
	#sigma_hat_inv = np.linalg.inv(data_train.iloc[:,0:n-1].cov().values)

	######## Calculate Prior probability ##########

	pi = np.array(data[label].value_counts()/m)

	######## Calculate Centroid ##########

	uk = np.array(data.groupby(label).mean())


	############ Calculate w & c ##############

	w = np.zeros([n-1,num_c])
	b = np.zeros((1,num_c))[0]

	for i in range(num_c):
		w[:,i] = sigma_hat_inv @ uk[i].T
		b[i] =  -1/2 * (uk[i,:].T @ sigma_hat_inv @ uk[i,:] + math.log(pi[i]))

	return w,b
	

def predict(w,b,X):

	output = np.dot(X,w)+b
	output = np.argmax(output, axis=1)

	return output


# set up data

iris = datasets.load_iris()
X = iris.data
y = iris.target
iris = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_length'])
iris['species'] = y
data_train,data_test = random_split(iris,portion = 0.8)

# test1
w,b = LDA_classifier(data_train,'species')
x_test = np.array(data_test.iloc[:,0:n-1])
y_test = np.array(data_test.iloc[:,n-1])
y_pred = predict(w,b,x_test)
confusion_matrix(y_test,y_pred)