import math
from math import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from numpy.random import random
from numpy.random import normal
import scipy
from scipy.stats import norm
#PT1: DEFINITION OF TEST PROBLEM

#DEFINING THE OBJECTIVE FUNCTION, I.E. THE FUCNTION WE WANT TO OPTIMIZE USING BAYESIAN OPT
def objective(x, noise = 0.1):
	noise = np.random.normal(loc = 0, scale = noise)
	return (x**2 + math.sin(x * math.pi * 5)**6.0) + noise
'''
#TEST OBJECTIVE FUNCTION BY DEFINING A SET OF X VALUES, INCREMENTING BY 0.01 FROM [0, 1]
X = np.arange(0, 1, 0.01)

#Y IS OUTPUT WHEN YOU PASS X THROUGH THE OBJECTIVE FUNTION
#NOTE: for true values, noise should be 0, which is why we pass 0 as noise during this test.
#NOTE: this is essentially called sampling the domain WITHOUT NOISE. 
y = [objective(x, 0) for x in X]

#NOTE: you can sample the domain WITH NOISE as well, as that's what's actually needed for BayesOpt
y_noise = [objective(x) for x in X]

#THE ENTIRE POINT OF BAYESIAN (OR ANY OTHER KIND OF) OPTIMIZATION IS TO FIND AN INPUT VALUE X FOR WHICH THE 
#GIVEN FUNCTION (OBJECTIVE FUNCTION) IS MAX, I.E. FINDING THE FUNCTION'S MAXIMA, OR A POINT AT WHICH THE FUNCTION'S
#VALUE IS MAXIMUM. DURING TYPICAL OPTIMIZATION PROBLEMS, WE WOULD NEVER KNOW THE TRUE OPTIMA DURING SOLVING THE PROBLEM, 
#HOWEVER, FOR TESTING, SINCE WE HAVE A TRUE SAMPLE SET (Y VALUES ABOVE) WE CAN EASILY FIND THE CORRESPONDING VALUE OF X
#FOR WHICH Y IS MAXIMUM. HOWEVER, THIS IS NOT KNOWN DURING NORMAL OPTIMIZATION, AND IS ESSENTIALLY WHAT WE WANT TO FIND OUT. 

ix = np.argmax(y)

#IX IS THEN THE LOCATION OF THE MAXIMUM VALUE OF OBJECTIVE FUNCTION

plt.scatter(X, y_noise)
plt.show()
plt.scatter(X, y)
plt.show()
'''

#DEFINING THE SURROGATE FUNCTION:
#THE SURROGATE FUNCTION IS A TECHNIQUE USED TO BEST APPROXIMATE THE MAPPING OF INPUT EXAMPLES TO AN OUTPUT SCORE. 
#A SURROGATE FUNCTION SUMMARIZES THE CONDITIONAL PROBABILITY OF AN OBJECTIVE FUNCTION 'f' GIVEN A SET OF DATAPOINTS 'D', 
# i.e. P(f|D)
#A GAUSSIAN PROCESS (GP) IS ONE OF THE BEST METHODS OF IMPLEMENTING A SURROGATE FUNCTION. IT CONSTRUCTS A JOINT PROBABILITY DISTRIBUTION 
# OVER THE VARIABLES ASSUMING A MULTIVARIATE GAUSSIAN PROCESS. DUE TO THIS, AN ESTIMATE FROM THE MODEL WILL BE A MEAN OF A DISTRIBUTION WITH A 
# STANDARD DEVIATION. 


#surrogate model approimation for the objective function
def surrogate(model, X):
	with catch_warnings():
		simplefilter('ignore')
		return model.predict(X, return_std = True)

def plot(X, y, model):
	plt.scatter(X, y)
	X_samples = np.asarray(np.arange(0, 1, 0.001))
	X_samples = X_samples.reshape(len(X_samples), 1)
	y_samples, _ = surrogate(model, X_samples)
	plt.plot(X_samples, y_samples)
	plt.show()

def aquisition(X, X_samples, model):
	yhat, _ = surrogate(model, X)
	best = max(yhat)

	mu, std = surrogate(model, X_samples)
	mu = mu[:, 0]

	prob = norm.cdf((mu - best)/(std + 1E-9))
	return prob

def opt_aquisition(X, y, model):
	X_samples = random(100)
	X_samples = X_samples.reshape(len(X_samples), 1)

	scores = aquisition(X, X_samples, model)
	ix = np.argmax(scores)
	return X_samples[ix, 0]

X = random(100)
y = np.asarray([objective(x) for x in X])

X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)


model = GaussianProcessRegressor()
model.fit(X, y)


X_act, y_act = X, y
model_initial = model

for i in range(100):
	#select next point to sample
	x = opt_aquisition(X, y, model)
	#sample the point
	actual = objective(x)
	est, _ = surrogate(model, [[x]])
	if i%100 == 0:
		print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
	#add this point to the dataset 
	X = np.vstack((X, [[x]]))
	y = np.vstack((y, [[actual]]))
	#update the model 
	model.fit(X, y)


#finally, plit all the samples and the final surrogate function
plot(X_act, y_act, model_initial)
plot(X, y, model)
#find the best result
ix = np.argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))




