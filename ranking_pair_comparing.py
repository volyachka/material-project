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

from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

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
    df = df.drop(['starting_structure', 'src_database', 'src_id'], axis = 1)
    df['group_id'] = 1
    df['relevance'] = -1
    df.sort_values(by='diffusion_mean_cm2_s', ascending=False, inplace = True)
    len_A = len(df[df['group'] == 'group_A'])
    len_B = len(df[df['group'] == 'group_B'])
    len_C = len(df[df['group'] == 'group_C'])
    len_E = len(df[df['group'] == 'group_E'])
    df.loc[df['group'] == 'group_A', 'relevance'] = np.arange(len_A)
    df.loc[df['group'] == 'group_B', 'relevance'] = len_A + np.arange(len_B)
    df.loc[df['group'] == 'group_E', 'relevance'] = len_A+len_B + np.arange(len_E)
    df.loc[df['group'] == 'group_C', 'relevance'] = len_A+len_B+len_E + np.arange(len_C)
    df['relevance'] = df.shape[0] - df['relevance']
    
    df.index = np.arange(0, len(df))

    group_names = np.array(np.zeros((len(df), len(df))), dtype=str)
    group_scores = np.array(np.zeros((len(df), len(df))))

    # group_names = np.loadtxt('ranking_pair_comparing/group_names.txt', dtype = str, delimiter=',')
    # group_scores = np.loadtxt('ranking_pair_comparing/group_scores.txt', dtype = int, delimiter=',')

    X, y = df[['fv_1p0_disconnected',
        'fv_1p0_connected',
        'fv_1p0_WARNlowPES',
        'fv_2p0_disconnected',
        'fv_2p0_connected',
        'fv_2p0_WARNlowPES',
        'fv_3p0_disconnected',
        'fv_3p0_connected',
        'fv_3p0_WARNlowPES',
        'fv_4p0_disconnected',
        'fv_4p0_connected',
        'fv_4p0_WARNlowPES',
        'barrier']], df['relevance'].astype(int).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            print(i, j)
            X_train = X_scaled[~np.isin(np.arange(len(X_scaled)), [i, j])]
            y_train = y[~np.isin(np.arange(len(y)), [i, j])]
            group_ids = df['group_id'][~np.isin(np.arange(len(df['group_id'])), [i, j])]
            train_pool = catboost.Pool(X_train, y_train, group_id = group_ids)

            X_test = X_scaled[np.isin(np.arange(len(X_scaled)), [i, j])]
            y_test = y[np.isin(np.arange(len(y)), [i, j])]
            group_ids = df['group_id'][np.isin(np.arange(len(df['group_id'])), [i, j])]
            test_pool = catboost.Pool(X_test, y_test, group_id = group_ids)



            model = catboost.CatBoostRanker(loss_function='PairLogitPairwise', verbose = False)
            model.fit(train_pool, plot=False)
            preds = model.predict(test_pool)
            preds = np.exp(preds) / (1 + np.exp(preds))
        
            first_group = df.iloc[i]['group'][6:]
            second_group = df.iloc[j]['group'][6:]
            first_score = df.iloc[i]['relevance']
            second_score = df.iloc[j]['relevance']

            group_names[i, j] = ''.join(sorted([first_group, second_group]))

            if (preds[0] >= preds[1] and first_score >= second_score) or (preds[1] >= preds[0] and second_score >= first_score):
                group_scores[i, j] = 1
            else:
                group_scores[i, j] = -1

            print(preds[0], preds[1])
            print(first_score, second_score)
            np.savetxt('ranking_pair_comparing/group_scores.txt', group_scores, fmt = '%d', delimiter=',')
            np.savetxt('ranking_pair_comparing/group_names.txt', group_names, fmt = '%s', delimiter=',')
            # with open("ranking_pair_comparing/group_names.txt", "w") as output:
            #     output.write(str(group_names))

            # with open("ranking_pair_comparing/group_scores.txt", "w") as output:
            #     output.write(str(group_scores))


