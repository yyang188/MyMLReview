import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.random.rand(10,2)
plt.scatter(data[:,0],data[:,1])
plt.show()


def kmeans(C,data):

	clu = True

	m = data.shape[0]
	n = data.shape[1]

	idx = np.random.choice(data.shape[0],C,replace=False)

	data = pd.DataFrame(data)
	data['label'] = [0]*m
	center = pd.DataFrame(data.iloc[idx,:])
	center = center.reset_index(drop = True)

	while clu:

		for i in range(0,m):

			obs = data.iloc[i,:]
			obs = np.array(obs[0:n])
			d = [0]*C

			for j in range(0,C): # Calculate the distance between a given point and each center
				c = np.array([center.iloc[j,0:n]]) 		
				d[j] =  np.sum(np.power(obs-c,2))				

			data.loc[i,'label'] = d.index(min(d)) # assign new label


		C_means = pd.DataFrame(data.groupby('label').agg('mean').values)# Calculating new means

		if center.equals(C_means):
			clu = False
		else:
			center = C_means

	return data

data = kmeans(2,data)
print(data)
plt.scatter(data.iloc[:,0],data.iloc[:,1],c = data['label'])
plt.show()












