import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 5):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    f = os.path.join(ROOT, 'data_batch_%d' % (5,))
    Xte, Yte = load_CIFAR_batch(f)
    return Xtr, Ytr, Xte, Yte


def load_models(models_dir):
    """
  Load saved models from disk. This will attempt to unpickle all files in a
  directory; any files that give errors on unpickling (such as README.txt) will
  be skipped.

  Inputs:
  - models_dir: String giving the path to a directory containing model files.
    Each model file is a pickled dictionary with a 'model' field.

  Returns:
  A dictionary mapping model file names to models.
  """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = pickle.load(f)['model']
            except pickle.UnpicklingError:
                continue
    return models


def save_model(path, model):
    """
  Use to save the model (NeuralNet object) into path. 
  
  Inputs: 
  - path: string with output path. 
  - model: NeuralNet object. 

  """
    with open(path, 'wb') as fp:
        pickle.dump(model, fp, protocol=2)
    
    return True


def load_model(path):
    """
  Use to load previously saved models. 

  Inputs: 
  - path: string with output path. 

  Returns: 
  NeuralNet object loaded. 
  """
    
    with open(path, 'rb') as fp:
        model = pickle.load(fp)
    return model
