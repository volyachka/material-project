import os
import io
import re

import numpy as np
import pandas as pd

from numpy import nan as Nan
from numpy import inf as inf
from tqdm import notebook as tqdm
from scipy.sparse import csr_matrix
from utils import load_csv

from pymatgen.core import Structure
from ase import units


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


def make_features(path):
    """
    Function to gather features from path folder into one dataframe.

    Parameters
    ----------
    path : str
        Path to folder with features.
        
    Returns
    -------
    files : list of str
        A list of pathes to the features. 
    
    feature_list : list
        list of features

    cnt_of_nan_features: list
        list of number of nans for all features
    """   
    
    files = sorted(os.listdir(path))

    feature_list = []
    cnt_of_nan_features = []
    name_of_features = []

    cnt_and_name_features = {}

    for file in files:
        features = np.load('{}/{}'.format(path, file), allow_pickle=True)
        feature_list.append(features)
        lost_features_count, _ = nan_and_inf_finder(features)

        name_of_features.append(file.split('_mode-structure', 1)[0])
        # print(file, name_of_features[-1], features.shape)

        cnt_and_name_features[name_of_features[-1]] = features.shape[1]

        # if lost_features_count != 0:
            # print("{} rows are lost in the feature: {}".format(lost_features_count, file))
        cnt_of_nan_features.append(lost_features_count)
    return name_of_features, feature_list, cnt_of_nan_features, cnt_and_name_features



def make_plane(labels_df, non_embedding_features):
    """
    From embeddings to plane features.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Path to folder with features.
        
    Returns
    -------
    labels_df : list of str
        A list of pathes to the features. 
    
    zero_columns : list
        list of features
    """   

    features = labels_df.drop(non_embedding_features, axis = 1).columns
    df_temp = pd.DataFrame()

    for _, feature in enumerate(features):
        data = labels_df[feature].to_numpy()
        data = np.array([np.array(x) for x in data]).T
        lenn = data.shape[0]
        columns = np.arange(lenn)
        columns = [str(x) + '_' + feature for x in columns]
        dictionary = dict(zip(columns, data))
        df = pd.DataFrame(dictionary)
        df_temp = pd.concat([df_temp, df], axis = 1)

    labels_df = pd.concat([df_temp, labels_df], axis = 1)
    labels_df = labels_df.drop(features, axis = 1)
    zero_columns = []

    for column in labels_df.columns:
        if labels_df[column].nunique() == 1:
            zero_columns.append(column)
    return labels_df, zero_columns



def merging_features(path_to_oxi_states, path_to_feature_folder, path_to_save):

    """
    merge embeddings to dataset of plane features

    Parameters
    ----------
    path_to_oxi_states : pd.DataFrame
        Path to dataset with oxi_states and material names.
        
    path_to_feature_folder: str
        Path to folder with features (.npy files)

    path_to_save: str
        Path to save csv dataframe

    Returns
    -------
    labels_df : list of str
        A list of pathes to the features. 
    
    nan_features : set
        set of features with Nans
    """   
        
    nan_features = set()
    labels_df = load_csv(path_to_oxi_states)
    files, feature_list, cnt_of_nan_features, cnt_and_name_features = make_features(path_to_feature_folder)
    for i, feature in enumerate(feature_list):
        # print(files[i])
        labels_df[files[i]] = feature.tolist()
        if cnt_of_nan_features[i] != 0:
            nan_features.add(files[i])
    labels_df.reset_index(drop=True, inplace=True)
    # save_csv(labels_df, path_to_save)
    return labels_df, nan_features, cnt_and_name_features




def dataset_preprocessing(df, structure_column):
    df[structure_column] = df[structure_column].apply(lambda x: Structure.from_str(x, 'json'))
    df['n_Li'] = df[structure_column].apply(lambda x: x.composition["Li"] / x.lattice.volume)

    D_to_sigma_factor = (
        df["n_Li"]  # 1 / A^3
            / (1000 * units.kB)  # eV
            * (1e24 / units.C)  # (A/cm)^3 * (e / C)
        )


    df.loc[df['diffusion_mean_cm2_s'] <= 1e-15, 'diffusion_mean_cm2_s'] = 1e-15

    df['sigma_S_cm'] = D_to_sigma_factor * df['diffusion_mean_cm2_s']
    df['sigma_S_cm_sem'] = D_to_sigma_factor * df['diffusion_sem_cm2_s']
    df['sigma_S_cm_err'] = df['sigma_S_cm_sem']

    df.loc[df['group'] == 'group_A', 'group'] = 'A'
    df.loc[df['group'] == 'group_B', 'group'] = 'B1'
    df.loc[df['group'] == 'group_E', 'group'] = 'B2'
    df.loc[df['group'] == 'group_C', 'group'] = 'C'

    return df

def get_featurizers_features_kahle():
    """
    Pull together features from matminer.featurizers to single dataset
    ----------
    """

    path_to_oxi_states = 'groups_and_oxi_states_5_frames/df_step_0.pkl'
    path_to_feature_folder = 'kahle_features'
    path_to_save = 'kahle.pkl'
    
    df_kahle, nan_features_kahle, cnt_and_name_features_kahle = merging_features(path_to_oxi_states, path_to_feature_folder, path_to_save)

    df_kahle = df_kahle.drop(['stru_traj', 'structure_A', 'structure_AM',
       'structure_CAN', 'structure_CAMN', 'structure_A40', 'structure_AM40',
       'structure_CAN40', 'structure_CAMN40'], axis = 1)

    df_kahle_plane, zero_columns_kahle = make_plane(df_kahle, ['stru_label', 'stru_id', 'group'])
    df_kahle_plane = df_kahle_plane.drop(zero_columns_kahle, axis = 1)

    for col in df_kahle_plane[df_kahle_plane.columns[df_kahle_plane.isna().any()].tolist()].isna().sum().to_frame().T.columns:
        df_kahle_plane[col].fillna((df_kahle_plane[col].mean()), inplace=True)

    df_kahle_plane = df_kahle_plane[df_kahle_plane['group'] != 'group_D']
    kahle = pd.read_csv('kahle.csv')
    df_kahle_fin = df_kahle_plane.merge(kahle, left_on = ['stru_label', 'stru_id', 'group'], right_on = ['stru_label', 'stru_id', 'group'])


    df_kahle_fin = dataset_preprocessing(df_kahle_fin, 'structure') 

    return df_kahle_fin, cnt_and_name_features_kahle




def get_featurizers_features_mpdb():
    """
    Pull together features from matminer.featurizers to single dataset
    ----------
    """

    path_to_oxi_states = 'mpdb/mpdb.pkl'
    path_to_feature_folder = 'mpdb_features'
    path_to_save = 'mpdb/structure_with_features.pkl'
    df_mpdb, nan_features_mpdb, cnt_and_name_features_mpdb = merging_features(path_to_oxi_states, path_to_feature_folder, path_to_save)

    df_mpdb = df_mpdb.drop(['nsites', 'elements', 'nelements',
        'volume', 'energy_above_hull', 'band_gap', 'database_IDs', 'structure_A', 'structure_AM', 'structure_CAN', 'structure',
        'structure_CAMN', 'structure_A40', 'structure_AM40', 'structure_CAN40', 'structure_CAMN40', 'fields_not_requested'], axis = 1)

    df_mpdb_plane, zero_columns_mpdb = make_plane(df_mpdb, ['formula_pretty', 'material_id'])



    path_to_oxi_states = 'groups_and_oxi_states_5_frames/df_step_0.pkl'
    path_to_feature_folder = 'kahle_features'
    path_to_save = 'kahle.pkl'
    
    df_kahle, nan_features_kahle, cnt_and_name_features_kahle = merging_features(path_to_oxi_states, path_to_feature_folder, path_to_save)
    df_kahle = df_kahle.drop(['stru_traj', 'structure_A', 'structure_AM',
        'structure_CAN', 'structure_CAMN', 'structure_A40', 'structure_AM40',
        'structure_CAN40', 'structure_CAMN40'], axis = 1)

    df_kahle_plane, zero_columns_kahle = make_plane(df_kahle, ['stru_label', 'stru_id', 'group'])


    df_mpdb_plane = df_mpdb_plane.drop(zero_columns_kahle, axis = 1)
    for col in df_mpdb_plane[df_mpdb_plane.columns[df_mpdb_plane.isna().any()].tolist()].isna().sum().to_frame().T.columns:
        df_mpdb_plane[col].fillna((df_mpdb_plane[col].mean()), inplace=True)

    return df_mpdb_plane, cnt_and_name_features_mpdb



def get_nn_features_kahle():
    df_barrier_features_kahle = pd.read_csv('datasets/exported.predictions.Kahle2020.v2.csv')

    barrier_robust_0p_features = list(filter(lambda x: x.find('barrier_robust_0p') != -1 and x.find('masked1p5') == -1, df_barrier_features_kahle.columns.to_list()))
    union_features = list(filter(lambda x: x.find('union') != -1 and x.find('masked1p5') == -1, df_barrier_features_kahle.columns.to_list()))
    df_barrier_features_kahle = df_barrier_features_kahle[barrier_robust_0p_features + union_features + ['src_id', 'diffusion_mean_cm2_s']]

    kahle = pd.read_csv('kahle.csv')
    df_kahle_fin = kahle.merge(df_barrier_features_kahle, left_on = ['src_id', 'diffusion_mean_cm2_s'], right_on = ['src_id', 'diffusion_mean_cm2_s'])
    df_kahle_fin = dataset_preprocessing(df_kahle_fin, 'structure')

    return df_kahle_fin


def get_nn_features_mpdb():
    df_barrier_features_mpdb = pd.read_csv('datasets/exported.predictions.mp.v2.csv')

    barrier_robust_0p_features = list(filter(lambda x: x.find('barrier_robust_0p') != -1 and x.find('masked1p5') == -1, df_barrier_features_mpdb.columns.to_list()))
    union_features = list(filter(lambda x: x.find('union') != -1 and x.find('masked1p5') == -1, df_barrier_features_mpdb.columns.to_list()))
    df_barrier_features_mpdb = df_barrier_features_mpdb[barrier_robust_0p_features + union_features + ['material_id']]

    return df_barrier_features_mpdb


from misc_utils.augment_preds import join_data_and_preds_exp, join_data_and_preds_icsd

def get_nn_features_exp():
    preds_mp = pd.read_csv("datasets/exported.predictions.mp.v2.csv")
    ref_mp = pd.read_csv("datasets/mp_Laskowski2023_map.csv")

    preds_mp_exp_initial = join_data_and_preds_exp(
        df_preds_full_mp=preds_mp,
        df_data_exp_mp=ref_mp,
    )

    preds_icsd_exp = join_data_and_preds_icsd(
        df_preds_icsd=pd.read_csv("datasets/exported.predictions.icsd.v3.csv"),
        df_data_exp_full=pd.read_csv("datasets/digitized_data_for_SSEs.csv"),
    )

    preds_mp_exp = pd.concat([preds_mp_exp_initial, preds_icsd_exp], axis=0).reset_index(drop=True)

    barrier_robust_0p_features = list(filter(lambda x: x.find('barrier_robust_0p') != -1 and x.find('masked1p5') == -1, preds_mp_exp.columns.to_list()))
    union_features = list(filter(lambda x: x.find('union') != -1 and x.find('masked1p5') == -1, preds_mp_exp.columns.to_list()))
    preds_mp_exp = preds_mp_exp[barrier_robust_0p_features + union_features + ['sample_weight', 'material_id']]

    return preds_mp_exp