import numpy as np


def _reluDeriv(x):
  x[x<=0] = 0
  x[x>0] = 1
  return x

def _sofmax(z):
  x=z.copy()
  # print(f"Z= {z.shape}")
  # exp_scores = np.exp(x)
  # sum_exp_scores = np.sum(exp_scores, axis=0)
  return np.exp(x) / np.exp(x).sum(axis=1) [:,None]

def _sofmaxDeriv(z):
  return np.multiply( z, 1 - z ) + sum(- z * np.roll( z, i, axis = 1 ) for i in range(1, z.shape[1] ))

def _Xentropy(Yp, Yr):
  N = len(Yr)
  y_hat = Yp[Yr.squeeze(), np.arange(N)]
  return np.sum(-np.log(y_hat)) / N
  # return (np.where(Yr==1,-np.log(Yp.clip(min=1e-8,max=None)), 0)).sum(axis=1)

def _XentropyDeriv(Yp, Yr):
  return np.where(Yr==1,-1/Yp, 0)


class ActivationFunction():
  sigm = (lambda z: 1 / (1 + np.e ** (-z)),
          lambda z: z * (1 - z))
  relu = (lambda z: np.maximum(0, z),
          lambda z: _reluDeriv(np.copy(z)))
  sofmax = (lambda z: _sofmax(z),
            lambda z: _sofmaxDeriv(z))

class CostFunction():
  mse = (lambda Yp, Yr: np.mean((Yp-Yr) ** 2),
         lambda Yp, Yr: (Yp - Yr))
  xe = (lambda Yp, Yr: _Xentropy(Yp, Yr),
        lambda Yp, Yr: _XentropyDeriv(Yp,Yr))