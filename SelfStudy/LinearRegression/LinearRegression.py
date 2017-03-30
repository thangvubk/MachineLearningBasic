from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
diabetes = datasets.load_diabetes()
print("raw data",diabetes.data.shape)

########################################################################
# Use Linear Regression for one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Concatenate biases
biases = np.ones(shape=(diabetes_X.shape[0],1))
diabetes_X = np.concatenate((diabetes_X, biases), axis = 1)

#Spit data to train set and test set
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#spit target to train and test sets
diabetes_Y_train = diabetes.target[:-20, np.newaxis]
diabetes_Y_test = diabetes.target[-20:, np.newaxis]

# Solve by normal equation
print('\n================SOLVE BY NORMAL EQUATION===================')
A = np.dot(diabetes_X_train.T, diabetes_X_train)
b = np.dot(diabetes_X_train.T, diabetes_Y_train)
w = np.dot(np.linalg.pinv(A),b)

# print result
print('Coefficients by normal equation:', w[0][0], w[1][0])
predict = np.dot(diabetes_X_test, w)
print('predict value and true value: \n', np.concatenate((predict, diabetes_Y_test), axis = 1))
print('Mean square error %.2f' %np.mean((predict - diabetes_Y_test) ** 2))
#plot output
plt.figure(1)
plt.plot(diabetes_X_train[:,0], diabetes_Y_train, 'bo', markersize=3)
plt.plot(diabetes_X_train[:,0], np.dot(diabetes_X_train, w), color='red')
plt.show ()

# solve by sklearn
print('\n================SOLVE BY SKLEARN===================')
regr = linear_model.LinearRegression(fit_intercept=False)

regr.fit(diabetes_X_train, diabetes_Y_train)

# print result
print('Coeffecients by sklearn:', regr.coef_)
print('Mean square error: %.2f' %np.mean((regr.predict(diabetes_X_test) - diabetes_Y_test)**2))

#solve by gradient decent
print('\n================SOLVE BY GRADIENT DESCENT===================')
alpha = 1
threshold = 1E-5
w = np.zeros(shape=(diabetes_X.shape[1],1))
def compute_cost():
	return np.sum((diabetes_Y_train - np.dot(diabetes_X_train, w))**2)/2/diabetes_X_train.shape[0]
cost = compute_cost()
cost_hist = [cost]
iter = 0
while(True):
	w = w - alpha*np.dot(diabetes_X_train.T,np.dot(diabetes_X_train, w) - diabetes_Y_train)/diabetes_X_train.shape[0]
	new_cost = compute_cost()
	cost_hist.append(new_cost)
	iter += 1
	print ('iter: %d %.2f' %(iter, new_cost))
	delta_cost = cost - new_cost
	cost = new_cost
	if(delta_cost < threshold):
		break
print('Coefficients by normal equation:', w[0][0], w[1][0])
predict = np.dot(diabetes_X_test, w)
print('predict value and true value: \n', np.concatenate((predict, diabetes_Y_test), axis = 1))
print('Mean square error %.2f' %np.mean((predict - diabetes_Y_test) ** 2))
plt.figure(2)
plot_X = [i for i in range(len(cost_hist))]
plt.plot(plot_X, cost_hist, color='black')
plt.show()


