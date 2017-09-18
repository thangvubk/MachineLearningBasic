import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2)
###############################
# Init data
###############################
meanA = [1, 1]
meanB = [3, 1]
cov = [[0.1, 0], [0, 2]]
N = 10
XA = np.random.multivariate_normal(meanA, cov, N).T
XB = np.random.multivariate_normal(meanB, cov, N).T


X = np.concatenate((XA, XB), axis=1)
X = np.concatenate((np.ones((1, 2*N)), X), axis=0)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis=1)

def h(w, x):
	return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):
	return np.array_equal(h(w, X), y)


def perceptron(X, y):
	d = X.shape[0]
	np.random.seed(1)
	w_init = np.random.randn(d, 1)
	w = [w_init]
	N = X.shape[1]
	mis_points = []
	iters = 0
	while True:
		# mix data
		iters += 1
		mix_id = np.random.permutation(N)
		for i in range(N):
			xi = X[:, mix_id[i]].reshape(d, 1)
			yi = y[0, mix_id[i]]
			if h(w[-1], xi)[0] != yi:
				mis_points.append(mix_id[i])
				w_new = w[-1] + yi*xi
				w.append(w_new)
		if has_converged(X, y, w[-1]):
			break
	return (w, mis_points, iters)

def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0:
        x11, x12 = -100, 100
        return plt.plot([x11, x12], [-(w1*x11 + w0)/w2, -(w1*x12 + w0)/w2], 'k')
    else:
        x10 = -w0/w1
        return plt.plot([x10, x10], [-100, 100], 'k')

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 
def viz_alg(w):
    it = len(w)    
    fig, ax = plt.subplots(figsize=(5, 5))  
    
    def update(i):
        ani = plt.cla()
        #points
        ani = plt.plot(XA[0, :], XA[1, :], 'b^', markersize = 8, alpha = .8)
        ani = plt.plot(XB[0, :], XB[1, :], 'ro', markersize = 8, alpha = .8)
        ani = plt.axis([-6 , 6, -6, 6])
        i2 =  i if i < it else it-1
        ani = draw_line(w[i2])
        if i < it-1:
            # draw one  misclassified point
            circle = plt.Circle((X[1, m[i]], X[2, m[i]]), 0.15, color='k', fill = False)
            ax.add_artist(circle)
        # hide axis 
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticks([])
        cur_axes.axes.get_yaxis().set_ticks([])

        label = 'PLA: iter %d/%d' %(i2, it-1)
        ax.set_xlabel(label)
        return ani, ax 
        
    anim = FuncAnimation(fig, update, frames=np.arange(0, it + 2), interval=1000)
    # save 
    #anim.save('pla_vis.gif', dpi = 100, writer = 'imagemagick')
    plt.show()
(w, m, iters) = perceptron(X, y)

viz_alg(w)