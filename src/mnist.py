import numpy as np

from libs.deep_learning import NeuralNetwork, Utils
from libs.functions import CostFunction
from libs.image_proccesing import get_nmist_dataset
from libs.nn_optimization import Optimizer, sgd
from libs.dataset_proccesing import createMinibatches

# MNIST path
IMAGES_PATH = './src/images/'


N_SAMPLES = 50000
INPUT_PARAMETERS = 28 * 28
EPOCHS = 1500
LR = 0.338

# Create Dataset
x_train_num, y_train_num, x_test_num, y_test_num = get_nmist_dataset(IMAGES_PATH)

X_TRAIN = x_train_num[:N_SAMPLES].reshape(N_SAMPLES, -1).astype(np.float32)/255
Y_TRAIN = y_train_num[:N_SAMPLES].reshape(N_SAMPLES, 1)
X_TEST = x_test_num.copy().reshape(10000, -1).astype(np.float32)/255
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
  for (x, y) in createMinibatches(X_TRAIN, Y_TRAIN, 500):
    pY = NN.run(x, OPTIMIZER, y)

  if(i % 10 == 0):
    loss.append(CostFunction.xe[0](pY, y))
    print(f"Epoch: {i} -> loss: {loss[-1]}")

NN.trainable = False

# Test the NN
print(f"X_TEST={X_TEST.shape}")
print(f"Y_TEST={Y_TEST.shape}")

pY = NN.run(X_TEST)

totalCost = 0
for i in range(pY.shape[1]):
  res = pY[:, i].reshape(10, 1)
  py = Y_TEST[:, i].reshape(1, 1)

  cost = CostFunction.xe[0](res, py)
  totalCost += cost

  print("----------------------------------------------------------------------")
  print(f"Propuesto: {res.argmax()}, Real: {py[0,0]}, Costo: {cost}")

totalCost /= len(pY)
print()
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print()
print(f"Medial Cost: {totalCost}")
print()
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

Utils.saveModel(NN, "./mnist")