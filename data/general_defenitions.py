import bz2
import _pickle as cPickle

def comp_pickle_save(data, filename):
    with bz2.BZ2File(filename, 'w') as f:
        cPickle.dump(data, f)