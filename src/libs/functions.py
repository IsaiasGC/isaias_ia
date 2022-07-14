import numpy as np


def _reluDeriv(z):
  z[z <= 0] = 0
  z[z > 0] = 1
  return z

def _sofmax(z):# z.shape = (clases, elements)
  max = np.max(z)
  exp_z = np.exp(z - max)
  sum_exp = np.sum(exp_z, axis=0)
  f_z = exp_z / sum_exp
  return f_z

def _sofmaxDeriv(z):  # z.shape = (clases, elements)
  return np.multiply( z, 1 - z ) + sum(- z * np.roll( z, i, axis = 1 ) for i in range(1, z.shape[1] ))

def _Xentropy(Yp, Yr):
  assert Yp.shape[1] == Yr.shape[1], 'Mismo numero de elementos en Yp(clases, elementos) y Yr(1,elementos)'
  y = Yr.copy()
  if (y.shape[0] != 1): y = y.argmax(axis=0)
  y = y.squeeze()
  m = Yp.shape[1]
  probs = Yp[y, np.arange(m)]
  log_y = -np.log(probs)
  f_Yp = np.sum(log_y) / m
  return f_Yp

def _XentropyDeriv(Yp, Yr):
  assert Yp.shape[1] == Yr.shape[1], 'Mismo numero de elementos en Yp(clases, elementos) y Yr(1,elementos)'
  y = Yr.copy()
  if (y.shape[0] != 1): y = y.argmax(axis=0)
  y = y.squeeze()
  m = Yp.shape[1]
  grad = Yp.copy()
  grad[y, np.arange(m)] -= 1
  grad = grad / m
  return grad


class ActivationFunction():
  sigm = (lambda z: 1 / (1 + np.e ** (-z)),
          lambda z: z * (1 - z))
  relu = (lambda z: np.maximum(0, z),
          lambda z: _reluDeriv(np.copy(z)))
  sofmax = (lambda z: _sofmax(z),
            lambda z: np.ones((z.shape)))

class CostFunction():
  mse = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
         lambda Yp, Yr: (Yp - Yr))
  xe = (lambda Yp, Yr: _Xentropy(Yp, Yr),
        lambda Yp, Yr: _XentropyDeriv(Yp, Yr))