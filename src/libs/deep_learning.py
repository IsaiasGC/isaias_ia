import numpy as np

from libs.functions import ActivationFunction

class NeuralLayer():
  def __init__(self, n_conn, n_neur, act_fun):
    self.b = np.zeros((n_neur, 1))
    self.w = (np.random.rand(n_neur, n_conn) / np.sqrt(n_conn / 2)) # Init parameters: Kaiming He
    self.act_fun = self.__getFunction(act_fun)

  def __getFunction(self, act_fun):
    if(act_fun == 'sigm'):
      return ActivationFunction.sigm
    elif (act_fun == 'relu'):
      return ActivationFunction.relu
    elif (act_fun == 'sofmax'):
      return ActivationFunction.sofmax
    else:
      return act_fun

class NeuralNetwork():
  def __init__(self, topology, act_funs):
    assert len(topology) == len(act_funs), 'Nuemero de Funciones no cuadra con numero de Neuronas'
    self.nn = self.__createLayers(topology, act_funs)
    self.topology = topology
    self.act_funs = act_funs
    self.trainable = False

  def __createLayers(self, topology, act_funs):
    nn = []
    lastLayer = topology[0][0]
    for l, layer in enumerate(topology):
      assert layer[0] == lastLayer, f'Numero de Entradas no cuadra en la Neurona: {l+1}'
      nn.append(NeuralLayer(layer[0], layer[1], act_funs[l]))
      lastLayer=layer[1]
      print(f"Layer: {l+1} -> W={nn[-1].w.shape}, b={nn[-1].b.shape}")

    return nn

  def run(self, x, optimizer = None, y = None):
    assert x.shape[0] == self.nn[0].w.shape[1], 'X debe de tener la forma (Input_Parameters, Zise_Batch)'

    z_a = [(None, x)] # (z, a)

    # Forward pass
    for l, layer in enumerate(self.nn):
      z = layer.w @ z_a[-1][1] + layer.b # Neuron value = Wx + b
      a = layer.act_fun[0](z) # Activate value = f(z)

      z_a.append((z, a))

    if self.trainable:
      assert optimizer != None, 'Debe proporcionar un Optimizer'
      assert x.shape[1] == y.shape[1], 'No hay mismo numero de Y(1, Zise_Batch) que de X(Input_Parameters, Zise_Batch)'
      optimizer.train(self.nn, y, z_a)
    
    return z_a[-1][1] # return S: output

class Utils():

  def saveModel(model, outfilename):
    '''Save model parameters to file
    Args:
    model (NeuralNetwork): the neural network model to save
    outfilename (str): absolute path to file to save model parameters.
    Parameters are saved using numpy.savez(), loaded using numpy.load().
    '''
    print('\n# <saveWeights>: Save network weights to file', outfilename)
    dump = {
              'topology': model.topology,
              'act_funs': model.act_funs
            }
    for l, nl in enumerate(model.nn):
      dump['W_%d' % l] = nl.w
      dump['b_%d' % l] = nl.b
    np.savez(outfilename, **dump)

    return

  def loadModel(abpathin):
    '''Load model parameters from file
    Args:
    abpathin (str): absolute path to file to load model parameters.
    Parameters are saved using numpy.savez(), loaded using numpy.load().
    '''
    print('\n# <saveWeights>: Load network weights from file', abpathin)
    with np.load(abpathin) as npzfile:
      topology = npzfile['topology']
      act_funs = npzfile['act_funs']
      NN = NeuralNetwork(topology, act_funs)
      for l, nl in enumerate(NN.nn):
          weights = npzfile['W_%d' % l]
          bias = npzfile['b_%d' % l]
          nl.w = weights
          nl.b = bias

    return NN