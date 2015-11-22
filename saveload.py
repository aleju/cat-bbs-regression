# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import h5py

def load_weights_seq(seq, filepath):
    """Loads the weights from an exactly specified weights file into a model.

    This function is identical to Kera's load_weights function, but checks first
    if the file exists and raises an error if that is not the case.

    In contrast to the load_weights function above, this one expects the full
    path to the weights file and does not search on its own for a well fitting one.

    Args:
        seq: The model for which to load the weights. The current weights
            will be overwritten.
        filepath: Full path to the weights file.
    """
    # Loads weights from HDF5 file
    if not os.path.isfile(filepath):
        raise Exception("Weight file '%s' does not exist." % (filepath,))
    else:
        f = h5py.File(filepath)
        for k in range(f.attrs['nb_layers']):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            seq.layers[k].set_weights(weights)
        f.close()
