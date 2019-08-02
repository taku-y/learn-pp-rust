from sklearn.datasets import make_moons
import numpy as np

X, Y = make_moons(n_samples = 30)
a = X[Y==0]
b = X[Y==1]

data = np.vstack([
    np.array(a), np.array(b)
])

np.savetxt("moon.csv", data, delimiter=",")
