import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

data = np.genfromtxt('training.csv', delimiter=',')
data = data[1:][:,1:]
plt.plot(data[:,0]) # training accuracy
plt.plot(data[:,2]) # testing accuracy
plt.legend(['Training','Testing'])
plt.title("Accuracy by Epoch")
plt.xlabel("epoch")
plt.ylim(0.0,1.0)
plt.show()
