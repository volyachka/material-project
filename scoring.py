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
from tqdm import tqdm
from scipy.sparse import csr_matrix
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_preprocessing import *
from sklearn.model_selection import KFold


def test_function_with_trajectories(dfs, number_of_folds, random_state):
    lenn =dfs[0].shape[0]
    indexes = np.array(lenn)
    preds_with_traj = np.zeros(lenn)
    preds_without_traj = np.zeros(lenn)
    kf = KFold(n_splits=number_of_folds, random_state=random_state, shuffle=True)
    kf.get_n_splits(indexes)
    for train_index, test_index in kf.split(indexes):
        assembled_pred = np.zeros(len(test_index))
        for i, df in enumerate(dfs):
            X, y = (df.drop(['is_good', 'stru_label', 'stru_id', 'barrier'], axis=1).to_numpy(), df['is_good'].astype(int).to_numpy())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            if i == 0:
                X_tr, X_te, y_tr, y_te = (X_scaled[train_index, :], X_scaled[test_index, :], y[train_index], y[test_index])
            else:
                X_tr, y_tr = (X_scaled[train_index, :], y[train_index])
            model = CatBoostClassifier(eval_metric='AUC', verbose = False)
            feature_names = ['F{}'.format(i) for i in range(np.array(X_tr).shape[1])]
            test_pool = Pool(np.array(X_te), y_te, feature_names=feature_names)
            summary = model.select_features(
                X = X_tr,
                y=y_tr,
                eval_set=test_pool,
                features_for_select= np.arange(len(X_tr[0])),
                num_features_to_select=50,
                steps=6,
                algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                shap_calc_type=EShapCalcType.Regular,
                train_final_model=True,
                logging_level='Silent',
                plot=False)
            y_pred = model.predict_proba(X_te)[:, 1]
            y_pred_tr = model.predict_proba(X_tr)[:, 1]
            print(f"For frame{i} roc auc score for train: {roc_auc_score(y_tr, y_pred_tr)}, for test {roc_auc_score(y_te, y_pred)}")
            if i == 0:
                preds_without_traj[test_index] = y_pred
        assembled_pred /= len(dfs)
        preds_with_traj[test_index] = assembled_pred
    return preds_without_traj, preds_with_traj, y
   

def test_function_without_trajectories(df, number_of_folds, random_state):
    X, y = (df.drop(['is_good', 'stru_label', 'stru_id', 'barrier'], axis=1).to_numpy(), df['is_good'].astype(int))
    preds = np.zeros(len(y))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=number_of_folds, random_state=random_state, shuffle=True)
    kf.get_n_splits(X_scaled)
    for index, (train_index, test_index) in enumerate(kf.split(X)):
        X_tr, X_te, y_tr, y_te = (X_scaled[train_index, :], X_scaled[test_index, :], y[train_index], y[test_index])
        model = CatBoostClassifier(eval_metric='AUC', verbose = False)
        feature_names = ['F{}'.format(i) for i in range(np.array(X_tr).shape[1])]
        test_pool = Pool(np.array(X_te), y_te, feature_names=feature_names)
        summary = model.select_features(
            X = X_tr,
            y=y_tr,
            eval_set=test_pool,
            features_for_select= np.arange(len(X_tr[0])),
            num_features_to_select=50,
            steps=6,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=True,
            logging_level='Silent',
            plot=False)
        y_pred = model.predict_proba(X_te)[:, 1]
        y_pred_tr = model.predict_proba(X_tr)[:, 1]
        preds[test_index] = model.predict_proba(X_te)[:, 1]
        print(f"roc auc score for train: {roc_auc_score(y_tr, y_pred_tr)}, for test {roc_auc_score(y_te, y_pred)}")
    return preds, y



def bootstrap(num_samples, array):
    bootstrap_means = np.zeros(num_samples)
    for i in range(num_samples):
        bootstrap_sample = np.random.choice(array, size=len(array), replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means[i] = bootstrap_mean
 
    estimated_mean = np.mean(bootstrap_means)
    estimated_std = np.std(bootstrap_means, ddof=1)
    
    return estimated_mean, estimated_std


def bootstrap_roc_auc(num_samples, y, preds):
    roc_aucs = np.zeros(num_samples)
    indexes = np.arange(len(y))
    for i in range(num_samples):
        bootstrap_indexes = np.random.choice(indexes, size=len(indexes), replace=True)
        roc_auc = roc_auc_score(y[bootstrap_indexes], preds[bootstrap_indexes])
        roc_aucs[i] = roc_auc
 
    estimated_mean = np.mean(roc_aucs)
    estimated_std = np.std(roc_aucs, ddof=1)
    
    return estimated_mean, estimated_std






def test_function_leave_one_out(dfs):
    lenn =dfs[0].shape[0]
    indexes = np.arange(lenn)
    preds_with_traj = np.zeros(lenn)
    preds_without_traj = np.zeros(lenn)
    kf = KFold(n_splits=lenn, shuffle=False)
    kf.get_n_splits(indexes)
    for iteration, (train_index, test_index) in enumerate(kf.split(indexes)):
        with open("results/progress.txt", "w") as output:
            output.write(str([iteration, test_index]))
        assembled_pred = np.zeros(len(test_index))
        for i, df in enumerate(dfs):
            X, y = (df.drop(['is_good', 'stru_label', 'stru_id', 'barrier'], axis=1).to_numpy(), df['is_good'].astype(int).to_numpy())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            if i == 0:
                X_tr, X_te, y_tr, y_te = (X_scaled[train_index, :], X_scaled[test_index, :], y[train_index], y[test_index])
            else:
                X_tr, y_tr = (X_scaled[train_index, :], y[train_index])
            model = CatBoostClassifier(eval_metric='AUC', verbose = False)
            feature_names = ['F{}'.format(i) for i in range(np.array(X_tr).shape[1])]
            test_pool = Pool(np.array(X_te), y_te, feature_names=feature_names)
            summary = model.select_features(
                X = X_tr,
                y=y_tr,
                eval_set=test_pool,
                features_for_select= np.arange(len(X_tr[0])),
                num_features_to_select=50,
                steps=6,
                algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                shap_calc_type=EShapCalcType.Regular,
                train_final_model=True,
                logging_level='Silent',
                plot=False)
            y_pred = model.predict_proba(X_te)[:, 1]
            y_pred_tr = model.predict_proba(X_tr)[:, 1]
            assembled_pred += y_pred
            print(f"For frame {i} roc auc score for train: {roc_auc_score(y_tr, y_pred_tr)}")
            if i == 0:
                preds_without_traj[test_index] = y_pred
                with open("results/leave_one_out_without_traj.txt", "w") as output:
                    output.write(str(preds_without_traj))
        assembled_pred /= len(dfs)
        preds_with_traj[test_index] = assembled_pred

        with open("results/leave_one_out_with_traj.txt", "w") as output:
            output.write(str(preds_with_traj))
    

    return preds_without_traj, preds_with_traj, y

    
   
