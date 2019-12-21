import math
from math import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

#DEFINING THE OBJECTIVE FUNCTION, I.E. THE FUCNTION WE WANT TO OPTIMIZE USING BAYESIAN OPT
def objective(x, noise = 0.1):
	noise = np.random.normal(loc = 0, scale = noise)
	return (x**2 + math.sin(x * math.pi * 5)**6.0) + noise

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


