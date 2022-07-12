import numpy as np

class Optimizer():
  def __init__(self, method, l2_cost, lr):
    self.optimization_method = method
    self.l2_cost = l2_cost
    self.lr = lr
  
  def train(self, nn, y, z_a):
    deltas = [] # dC/dz: El cambio del Coste con respecto del valor de la neurona

    #Backward pass
    for l in reversed(range(0, len(nn))):
      a = z_a[l+1][1] # Activate value

      if l == len(nn) - 1:
        deltas.insert(0, self.l2_cost[1](a, y) * nn[l].act_fun[1](a))
      else:
        deltas.insert(0, _W.T @ deltas[0] * nn[l].act_fun[1](a))

      _W = nn[l].w

      # Calculate correction
      self.optimization_method(nn[l],deltas[0], z_a[l][1], self.lr)

def sgd(layer, deltas, a, lr):
    layer.b -= np.mean(deltas, axis=1, keepdims=True) * lr
    layer.w -= deltas @ a.T * lr