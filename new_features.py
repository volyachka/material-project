import pandas as pd
from utils import load_csv
import os
import pickle
import numpy as np
import pandas as pd
from numpy import nan as Nan
from numpy import inf as inf
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_predict
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectFromModel

from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from data_preprocessing import *
from scoring import bootstrap_roc_auc
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from tqdm import tqdm
from utils import *
from scoring import * 

if __name__ == "__main__":

    df = pd.read_csv('exported.predictions.Kahle2020.csv')
    groups = load_csv(f"groups_and_oxi_states_5_frames/df_features_step_0.pkl")
    df = df.merge(groups, left_on=['src_id'], right_on=['stru_id'])
    df = df.drop(['group', 'starting_structure', 'src_database', 'src_id', 'diffusion_mean_cm2_s'], axis = 1)
    X, y = df.drop(['stru_label', 'stru_id', 'is_good', 'fv_1p0_WARNlowPES', 'fv_2p0_WARNlowPES', 'fv_3p0_WARNlowPES', 'fv_4p0_WARNlowPES'], axis = 1), df['is_good'].astype(int).to_numpy()
    feature_importance = np.zeros(X.shape[1])
    is_top_50 = np.zeros(X.shape[1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    loo = LeaveOneOut()
    loo.get_n_splits(X_scaled)
    preds = np.zeros(len(y))
    with open("new_features_result/feature_names.txt", "w") as output:
        output.write(str(X.columns.to_list()))

    with open("new_features_result/labels.txt", "w") as output:
        output.write(str(y))

    
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        print(i)
        X_tr, X_te, y_tr, y_te = (X_scaled[train_index, :], X_scaled[test_index, :], y[train_index], y[test_index])
        model = CatBoostClassifier(eval_metric='AUC', verbose = False, random_state = 42)
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
        preds[test_index] = model.predict_proba(X_scaled[test_index])[:, 1]

        is_top_50[summary['selected_features']] += 1
        feature_importance += model.get_feature_importance()

        with open("new_features_result/preds.txt", "w") as output:
            output.write(str(preds))

        with open("new_features_result/is_top_50.txt", "w") as output:
            output.write(str(is_top_50))

        with open("new_features_result/feature_importance.txt", "w") as output:
            output.write(str(feature_importance))

    estimated_mean, estimated_std = bootstrap_roc_auc(1000, y, preds)
    print(estimated_mean, estimated_std)
