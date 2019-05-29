import numpy as np
import pandas as pd
import math
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from itertools import groupby


# Randomly_split dataset

def random_split(data,portion = 0.8):
	m = data.shape[0]
	n = data.shape[1]
	idx =  np.random.choice(m,m,replace = False)  
	data_train = data.iloc[idx[0:np.int(round(m*portion))],:]
	data_test = data.iloc[idx[np.int(round(m*portion)):],:]

	return data_train,data_test

def most_frequent(List): 
    max_freq = 0
    ele = List[0] 
      
    for i in List: 
        freq = List.count(i) 
        if(freq> max_freq): 
            max_freq = freq
            ele = i 
  
    return ele

# train: dataset with label
# test: test dataset without label
# nearest: # of neighbor to look at

def KNN(train,test,nearest,m,n):
	X = np.array(train.iloc[:,0:n-1])
	label = list(train.iloc[:,n-1])
	c = [0]*m

	for i in range(m):
		diff = X-test[i,:]
		diff = diff @ diff.T
		diff = list(np.diag(diff))
		vote = [ c for _,c in sorted(zip(diff,label))]
		vote = vote[0:nearest]
		c[i] = most_frequent(vote)

	return c
		


# set up data

iris = datasets.load_iris()
X = iris.data
y = iris.target
iris = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_length'])
iris['species'] = y
train,test = random_split(iris,portion = 0.8)


m = test.shape[0]
n = test.shape[1]
test_c = np.array(test.iloc[:,n-1])
test = np.array(test.iloc[:,0:n-1])
pred = KNN(train,test, 3,m,n)
confusion_matrix(test_c,pred)