import numpy as np

from sklearn.datasets import make_circles
from libs.deep_learning import Utils
from libs.functions import CostFunction

N_SAMPLES = 1000
INPUT_PARAMETERS = 2

# Create Dataset
X, Y =make_circles(n_samples=N_SAMPLES, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis] # Add new axis to Y

X = X.T
Y = Y.T

# Create NN
NN = Utils.loadModel("./models/make_circles.npz")

for i, res in enumerate(NN.run(X).squeeze()):
  print("----------------------------------------------------------------------")
  print(f"Propuesto: {res}, Real: {Y[:,i]}, Costo: {CostFunction.mse[0](res, Y[:,i])}")