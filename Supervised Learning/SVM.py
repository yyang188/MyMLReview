from sklearn import datasets
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
	
iris = datasets.load_iris()
X = iris.data
y = iris.target
iris = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_length'])
iris['species'] = y

# Since Iris dataset has three classes, here I will remove one of the class.
reducedIris = iris[iris['species']!=2]

#################################



learning_rate = 0.1
repeat_time = 1000


# Define Dataset

n_x = len(reducedIris .columns) - 1
m = len(reducedIris)
X = X[0:m,:]
y = y[0:m]

# random initialization

def random_initialization(n_x,k = 10):

	w = np.random.randn(1,n_x)*k
	b = np.random.randn(1)*k

	parameters= {'w':w,'b':b}

	return parameters


# define the cose function

def compute_cost(w,b,X,y,learning_rate = 0.1):

	m = len(X)

	One_yFx = (1-y * (np.dot(w,X.T) + b)).tolist()[0]
	cost = np.array(list(map(lambda x: max(0,x),One_yFx)))
	cost = sum(cost)/m + learning_rate*np.dot(w,w.T)/2
	return cost


#parameters = random_initialization(n_x)
#w = parameters['w']
#b = parameters['b']
#cost = compute_cost(w,b,X,y,learning_rate = 0.1)
#print(cost)


# gradients

def svmSGD(w,b,X,y,learning_rate = 0.5,lam = 0.1,epoch = 10000):

	n_x = X.shape[1]
	m = X.shape[0]

	for i in range(epoch):

		random_idx = np.random.choice(m,m,replace = False)

		for j,idx in enumerate(random_idx):

			yFx = (y[idx] * (np.dot(w,X[idx,:].T) + b))[0]

			if yFx >= 1:
				dw = lam*w
				db = 0
			else:
				dw = (lam*w - y[idx]*X[idx,:])
				db = (-y[idx])

			w = w - learning_rate*dw
			b = b - learning_rate*db

	return w,b


# gradients(w,b,X,y,learning_rate = 0.1,lam = 0)

def predict(w,b,X):

	 output = list((y * (np.dot(w,X.T) + b))[0])
	 output = [1 if ot >= 1 else -1 for ot in output]

	 return np.array(output)






