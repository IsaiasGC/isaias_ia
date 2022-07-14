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
LR = 0.003

# Create Dataset
x_train_num, y_train_num, _, _ = get_nmist_dataset(
    IMAGES_PATH)

x_train_num = x_train_num[N_SAMPLES:].reshape(10000, -1).astype(np.float32)/255
y_train_num = y_train_num[N_SAMPLES:].reshape(10000, 1)

X_VAL = x_train_num.T
Y_VAL = y_train_num.T

print(f"X_VAL={X_VAL.shape}")
print(f"Y_VAL={Y_VAL.shape}")

# Create NN
NN = Utils.loadModel('./models/mnist.npz')

totalCost = Utils.accuracy(NN, X_VAL, Y_VAL)
print()
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print()
print(f"Acurracy: {totalCost}")
print()
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
