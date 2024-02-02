import os
import io
import sys
import re
import time
import math
import pickle

import numpy as np
import pandas as pd

from numpy import nan as Nan
from numpy import inf as inf
from tqdm import notebook as tqdm
from scipy.sparse import csr_matrix

def nan_and_inf_finder(features):
    """
    Function to find the Nan, Null, or Inf values in the feature dataframe.

    Parameters
    ----------
    features : np.array
        A feature representation for each structure
        
    Returns
    -------
    lost_features_count : int
        A count of all rows that contain errors. 
    
    valid_features : np.array()
        The index positions for the valid features
    """      
    nan_array = np.isnan(features).any(1)
    inf_array = np.isinf(features).any(1)
    lost_features_count = np.logical_or(nan_array, inf_array).sum()
    valid_features = np.logical_not(np.logical_or(nan_array, inf_array))
    return lost_features_count, valid_features

def nan_and_inf_finder_SOAP(features):
    """
    Function to find the Nan, Null, or Inf values in the SOAP feature represntation.
    Because of SOAP's immense size, it is always saved as a sparse matrix. Thus this
    function is required to specifically handle SOAP. 

    Parameters
    ----------
    features : scipy.sparse.csr.csr_matrix
        A feature representation for each structure
        
    Returns
    -------
    lost_features_count : int
        A count of all rows that contain errors. 
    
    valid_features : np.array
        The index positions for the valid features
    """     
    if np.isnan(features.data).any() == False:
        if np.isinf(features.data).any() == False:
            lost_features_count = 0
            valid_features = np.ones(np.shape(features)[0], dtype=bool)
            return lost_features_count, valid_features
        


def save_sparse_features(features, filename):
    """
    Function to save a sparse feature representation for each feature. The files are saved with the same name
    but in a new directory: 'saved_sparse_features'.

    Parameters
    ----------
    features : np.array
        A feature representation for each structure.
        
    filename: str
        The original filename for the feature. 
    """ 
    sparse_features = csr_matrix(features)
    
    # save the sparse representation
    save_path = os.path.join(os.getcwd(), 'sparse_features/{}.pkl'.format(filename))
    save_file = open(save_path, 'wb')
    pickle.dump(sparse_features, save_file)
    save_file.close()
    return sparse_features