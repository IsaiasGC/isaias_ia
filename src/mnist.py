import numpy as np

from libs.deep_learning import NeuralNetwork, Utils
from libs.functions import CostFunction
from libs.image_proccesing import get_nmist_dataset
from libs.nn_optimization import Optimizer, sgd
from libs.dataset_proccesing import createMinibatches

# MNIST path
IMAGES_PATH = './src/images/'


N_SAMPLES = 5000
INPUT_PARAMETERS = 28 * 28
EPOCHS = 100
LR = 0.3

# Create Dataset
x_train_num, y_train_num, x_test_num, y_test_num = get_nmist_dataset(IMAGES_PATH)

X_TRAIN = x_train_num[:N_SAMPLES].reshape(N_SAMPLES, -1).astype(np.float32)/255
Y_TRAIN = y_train_num[:N_SAMPLES].reshape(N_SAMPLES, 1)
# x_val = x_train_num[N_SAMPLES:].reshape(10000, -1).astype(np.float)/255
# y_val = y_train_num[N_SAMPLES:].reshape(10000, 1)
X_TEST = x_test_num.copy().reshape(10000, -1).astype(np.float)/255
Y_TEST = y_test_num.copy().reshape(10000, 1)

X_TRAIN = X_TRAIN.T
Y_TRAIN = Y_TRAIN.T
X_TEST = X_TEST.T
Y_TEST = Y_TEST.T

print(f"X_TRAIN={X_TRAIN.shape}")
print(f"Y_TRAIN={Y_TRAIN.shape}")

# Create NN
topology = [(INPUT_PARAMETERS, 200), (200, 10)]
act_funs = ["relu", "sofmax"]
NN = NeuralNetwork(topology, act_funs)

# Create Optimizer
OPTIMIZER = Optimizer(sgd, CostFunction.xe, LR)

# resolution = 100
loss = []

# Train the NN
NN.trainable = True

for i in range(EPOCHS):
  for (x, y) in createMinibatches(X_TRAIN, Y_TRAIN, 1000):
    pY = NN.run(x, OPTIMIZER, y)

  # if(i % 10 == 0):
  loss.append(CostFunction.xe[0](pY, y))
  print(f"Epoch: {i} -> loss: {loss[-1]}")
NN.trainable = False

# Test the NN
print(f"X_TEST={X_TEST.shape}")
print(f"Y_TEST={Y_TEST.shape}")

yP = NN.run(X_TEST).squeeze()
totalCost = 0
for i, res in enumerate(yP):
  cost = CostFunction.xe[0](res, Y_TEST[:,i])
  totalCost += cost
  print("----------------------------------------------------------------------")
  print(f"Propuesto: {res}, Real: {Y_TEST[:,i]}, Costo: {cost}")

totalCost /= len(yP)
print(f"Medial Cost: {totalCost}")

Utils.saveModel(NN, "./make_circles")