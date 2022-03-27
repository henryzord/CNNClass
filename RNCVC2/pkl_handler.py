def save_obj(obj, name):
    import cPickle as pickle
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_features(path):
    import cPickle as pickle
    with open(path, 'rb') as f:
        return pickle.load(f)

