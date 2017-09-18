import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(-6, 6, 1024)

plt.ylim(-.5, 3)
plt.plot(X, np.sinc(X), c = 'k')
plt.show()