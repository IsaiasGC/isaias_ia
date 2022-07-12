import numpy as np

from sklearn.datasets import make_circles
from libs.deep_learning import NeuralNetwork, Utils
from libs.functions import CostFunction
from libs.nn_optimization import Optimizer, sgd
from libs.dataset_proccesing import makeDatase

N_SAMPLES = 1000
INPUT_PARAMETERS = 2
EPOCHS = 3000
LR = 0.03

# Create Dataset
X, Y =make_circles(n_samples=N_SAMPLES, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis] # Add new axis to Y

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = makeDatase(X.T, Y.T)

print(f"X_TRAIN={X_TRAIN.shape}")
print(f"Y_TRAIN={Y_TRAIN.shape}")

# Create NN
topology = [(INPUT_PARAMETERS, 16), (16, 8), (8, 1)]
act_funs = ["relu", "relu", "sigm"]
NN = NeuralNetwork(topology, act_funs)

# Create Optimizer
OPTIMIZER = Optimizer(sgd, CostFunction.mse, LR)

# resolution = 100
loss = []

# Train the NN
NN.trainable = True
for i in range(EPOCHS):
  pY = NN.run(X_TRAIN, OPTIMIZER, Y_TRAIN)

  if(i % 10 == 0):
    loss.append(CostFunction.mse[0](pY, Y_TRAIN))
    print(f"Epoch: {i + 1} -> loss: {loss[-1]}")

    # _x0 = np.linspace(-1.5, 1.5, resolution)
    # _x1 = np.linspace(-1.5, 1.5, resolution)
    # _y = np.zeros((resolution, resolution))

    # for i0, x0 in enumerate(_x0):
    #   for i1, x1 in enumerate(_x1):
    #     _y[i0,i1] = NN.run(np.array([[x0, x1]]), Y, CostFunction.mse, 0.03)[0][0]
NN.trainable = False

# Test the NN
print(f"X_TEST={X_TEST.shape}")
print(f"Y_TEST={Y_TEST.shape}")

yP = NN.run(X_TEST).squeeze()
totalCost = 0
for i, res in enumerate(yP):
  cost = CostFunction.mse[0](res, Y_TEST[:,i])
  totalCost += cost
  print("----------------------------------------------------------------------")
  print(f"Propuesto: {res}, Real: {Y_TEST[:,i]}, Costo: {cost}")

totalCost /= len(yP)
print(f"Medial Cost: {totalCost}")

Utils.saveModel(NN, "./make_circles")