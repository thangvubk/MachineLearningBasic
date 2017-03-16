from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt

#height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183 ]]).T
#weight (kg)
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
#visualize data
plt.plot(X, y, 'ro')
plt.axis([140, 190, 45, 75])
plt.xlabel('Height(cm)')
plt.ylabel('Weight(Kg)')
#plt.show()

#build Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

#calculating weight of fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print ('w = ', w)

#prepareing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0

plt.plot(X.T, y.T, 'ro')
plt.plot(x0, y0)
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

testWeight1 = 155
testWeight2 = 160

y1 = w_1*testWeight1 + w_0
y2 = w_1*160 + w_0

print (u'Predicted weight of person with height 155cm: %.2f (kg), real number 52(kg)' %(y1))
print (u'Predicted weight of person with height 160cm: %.2f (kg), real number 56(kg)' %(y2))