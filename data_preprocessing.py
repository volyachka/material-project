import numpy as np
import pandas as pd
from numpy import nan as Nan
from numpy import inf as inf
from tqdm import notebook as tqdm
from scipy.sparse import csr_matrix
import os
import io
import sys
import re
import time
import math
import pickle
from sklearn.preprocessing import StandardScaler


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

def load_csv(path):
    save_path = os.path.join(os.getcwd(), path)
    open_file = open(save_path, 'rb')
    labels_df = pickle.load(open_file)
    open_file.close()
    return labels_df

def save_csv(labels_df, path):
    save_path = os.path.join(os.getcwd(), path)
    save_file = open(save_path, 'wb')
    pickle.dump(labels_df, save_file)
    save_file.close()

def make_features(path, index):
    valid_features_df = pd.DataFrame()
    files = os.listdir(path)
    feature_list = list()
    cnt_of_nan_features = list()
    print('Features for {}'.format(index))
    for file in files:
        # remove the .npy extension
        filename = file[0:-4]
        if re.search('SOAP', file):
            features = csr_matrix(np.load(io.BytesIO(open('{}/{}'.format(path, file), 'rb').read()), allow_pickle=True).all())
            lost_features_count, valid_features = nan_and_inf_finder_SOAP(features)
            # save the sparse representation
            save_path = os.path.join(os.getcwd(), 'groups_and_oxi_states/df_step_{}/sparse_features/{}.pkl'.format(index, filename))
            save_file = open(save_path, 'wb')
            pickle.dump(features, save_file)
            save_file.close()
        elif re.search('ipynb_checkpoints', file):
            next
        else:
            features = np.load('{}/{}'.format(path, file), allow_pickle=True)
            feature_list.append(features)
            lost_features_count, valid_features = nan_and_inf_finder(features)
            # create a sparse representation for each feature
            sparse_features = save_sparse_features(features, filename)
            # feature_list.append(sparse_features)
        valid_features_df[filename] = valid_features
        if lost_features_count != 0:
            print("{} rows are lost in the feature: {}".format(lost_features_count, file))
        cnt_of_nan_features.append(lost_features_count)
    return feature_list, cnt_of_nan_features


def make_plane(labels_df):
  features = labels_df.drop(['stru_label', 'group'], axis = 1).columns
  df_t = pd.DataFrame()
  last_index = 0
  for i, feature in enumerate(features):
      data = labels_df[feature].to_numpy()
      data = np.array([np.array(x) for x in data]).T
      lenn = data.shape[0]
      columns = np.arange(last_index, last_index + lenn)
      last_index += lenn
      dictionary = dict(zip(columns, data))
      df = pd.DataFrame(dictionary)
      df_t = pd.concat([df_t, df], axis = 1)

  labels_df = pd.concat([df_t, labels_df], axis = 1)
  labels_df = labels_df.drop(features, axis = 1)
  zero_columns = list()
  for column in labels_df.columns:
    if labels_df[column].nunique() == 1:
        zero_columns.append(column)
  return labels_df, zero_columns


def feature_preprocessing(dfs, include_trajectories = False):
    grand_X = list()
    grand_y = list()
    for df in dfs:
        scaler = StandardScaler()
        X, y = (df.drop(['is_good', 'stru_label'], axis=1), df['is_good'])
        X_scaled = scaler.fit_transform(X)
        grand_X.append(X_scaled)
        y = y.astype(int)
        grand_y.append(y)
        if not include_trajectories:
            break
    return grand_X, grand_y

def get_train_test(grand_X, grand_y, random_split):
    X_tr = list()
    y_tr = list()
    X_te = list()
    y_te = list()
    for X_scaled, y in zip(grand_X, grand_y):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=random_split)
        X_tr.extend(X_train)
        X_te.extend(X_test)
        y_tr.extend(y_train)
        y_te.extend(y_test)
    X_tr = np.array(X_tr)
    X_te = np.array(X_te)
    return X_tr, X_te, y_tr, y_te