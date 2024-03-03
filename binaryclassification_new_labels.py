import pandas as pd
from utils import load_csv
import numpy as np
import os
import pickle
import numpy as np
import pandas as pd
from numpy import nan as Nan
from numpy import inf as inf
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_predict
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm

from catboost import CatBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from data_preprocessing import *
from scoring import bootstrap_roc_auc
from catboost import CatBoostRanker, Pool
from copy import deepcopy
import numpy as np
import os
import pandas as pd 
import catboost


if __name__ == "__main__":

    df = pd.read_csv('exported.predictions.Kahle2020.csv')
    groups = load_csv(f"groups_and_oxi_states_5_frames/df_features_step_0.pkl")
    df = df.merge(groups, left_on=['src_id'], right_on=['stru_id'])
    df = df.drop(['starting_structure', 'src_database', 'src_id', 'diffusion_mean_cm2_s'], axis = 1)
    df.drop_duplicates(subset=['stru_id', 'stru_label'], keep='first', inplace=True, ignore_index=False)
    data = pd.read_csv('fv.v2.Kahle2020.csv')
    data = data[data['temperature'] == 1000]
    df = df.merge(data, left_on=['stru_id'], right_on=['src_id'])
    df.drop_duplicates(subset=['stru_id', 'stru_label'], keep='first', inplace=True, ignore_index=False)

    df['is_good'] = False
    df.loc[(df['group'] == 'group_A') | (df['stru_id'].isin([1008693, 39761, 421083]) == True), 'is_good'] = True

    df = df.drop([
        'group',
        'stru_label',
        'stru_id',
        'temperature',
        'diffusion_std_cm2_s',
        'diffusion_sem_cm2_s',
        'label',
        'first_frame_structure',
        'starting_structure',
        'diffusion_mean_cm2_s',
        'src_database',
        'fv_1p0_WARNlowPES',
        'fv_2p0_WARNlowPES',
        'fv_3p0_WARNlowPES',
        'fv_4p0_WARNlowPES',
        'src_id',], axis = 1)
    

    X, y = (df.drop(['is_good'], axis=1).to_numpy(), df['is_good'].astype(int).to_numpy())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lenn = len(y)
    indexes = np.arange(lenn)
    preds = np.zeros((lenn, 2))
    classes = np.zeros((lenn, 2), dtype=str)
    kf = KFold(n_splits=lenn, shuffle=False)
    kf.get_n_splits(indexes)
    for iteration, (train_index, test_index) in enumerate(kf.split(indexes)):
        print(iteration)
        X_tr, X_te, y_tr, y_te = (X_scaled[train_index, :], X_scaled[test_index, :], y[train_index], y[test_index])
        model = CatBoostClassifier(eval_metric='AUC', verbose = False)
        feature_names = ['F{}'.format(i) for i in range(np.array(X_tr).shape[1])]
        summary = model.select_features(
            X = X_tr,
            y=y_tr,
            features_for_select= np.arange(len(X_tr[0])),
            num_features_to_select=50,
            steps=6,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=True,
            logging_level='Silent',
            plot=False)
        y_pred = model.predict_proba(X_te)
        preds[test_index] = y_pred
        classes[test_index] = model.classes_
        np.savetxt('binaryclassification_results_new_labels/preds.txt', preds, fmt='%.4e', delimiter=' ', newline='\n')
        np.savetxt('binaryclassification_results_new_labels/classes.txt', classes, fmt='%s', delimiter=' ', newline='\n')