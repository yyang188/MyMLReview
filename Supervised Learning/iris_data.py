def iris():

	from sklearn import datasets
	import pandas as pd
	import numpy as np
	
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target
	iris = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_length'])
	iris['species'] = y
	return iris

