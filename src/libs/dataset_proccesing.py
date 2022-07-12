import numpy as np

def makeDatase(x, y, test = 0.2, shuffle = True):
  assert x.shape[1] == y.shape[1], 'No hay mismo numero de Y(1, Num_Element) que de X(Input_Parameters, Num_Element)'
  total_data = x.shape[1]
  num_test = int(total_data * test)
  num_train = total_data - num_test
  print(f"Total: {total_data}, Train: {num_train}, Test: {num_test} ")
  if shuffle: 
      idxs = np.arange(total_data)
      np.random.shuffle(idxs)
      x = x[:,idxs]
      y = y[:,idxs]
  return (x[:,:num_train], x[:,num_train+1:], y[:,:num_train], y[:,num_train+1:])


def createMinibatches(x, y, mb_size, shuffle = True):
  assert x.shape[1] == y.shape[1], 'No hay mismo numero de Y(1, Num_Element) que de X(Input_Parameters, Num_Element)'
  total_data = x.shape[1]
  if shuffle: 
      idxs = np.arange(total_data)
      np.random.shuffle(idxs)
      x = x[:,idxs]
      y = y[:,idxs]
      
  return ((x[:,i:i+mb_size], y[:,i:i+mb_size]) for i in range(0, total_data, mb_size))