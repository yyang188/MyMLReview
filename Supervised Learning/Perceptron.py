import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

class Perceptron(object):

	def __init__(self,rate =0.01,iter_n = 100):
		self.rate = rate
		self.iter = iter_n

	def fit(self,X,y):

		self.w = np.zeros(X.shape[1])
		self.b = np.zeros(1)
		self.miss = []

		for i in range(self.iter):

			total_error = 0

			for j in range(X.shape[0]):

				error = y[j]- np.where( X[j,:]@self.w.T+self.b >= 0,1,0 )
				update = self.rate*error
				self.w = self.w+update*X[j,:]
				self.b = self.b+update

				total_error += abs(error)

			self.miss.append(total_error)

		return self


X = X[0:100,:]
y = y[0:100]
Perceptron_classifier = Perceptron(iter_n=10)

classifier = Perceptron_classifier.fit(X,y)

plt.plot(classifier.miss)
plt.xlabel('Epochs')
plt.ylabel('Number of misclassfication')
plt.show()