import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from utils import bootstrap_roc_auc

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from catboost import CatBoostClassifier, Pool

from training_functions_backup import train_loop, calculate_ROClikeComparisonMetricsKahle, plot_distribution_compared
from tqdm import trange
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV

class RandomModel():
    def __init__(self, random_state):
        self.random_state = random_state
    def fit(self, X, labels, sample_weight):
        pass
    def predict_proba(self, X):
        probability = np.random.rand(X.shape[0])
        return np.stack([probability, 1 - probability], axis=1)
    
class ModelEvaluation():
    """
    This class is designed to test the models and pipelines
    """

    def __init__(self, 
                 df_kahle_fin: pd.DataFrame, 
                 preds_Kahle2020: pd.DataFrame, 
                 model_name: str, 
                 params: dict,
                 ):

        """
        Parameters
        ----------

        df_kahle_fin: pd.DataFrame
            Pandas Dataframe with columns:  diffusion_mean_cm2_s [cm^2/s],
                                            sigma_S_cm [S/cm]
                                            group [A, B1, B2, C]
        model_name: str
            Available names: 'catboost', 'lightgbm', 'xgboost'

        params: dict
            Params of the model, like learning rate or max_depth
        """

        self.model_name = model_name
        self.df_kahle_fin = df_kahle_fin
        self.preds_Kahle2020 = preds_Kahle2020
        self.params = params

        self.roc_auc_averaged = []
        self.roc_like_comparison_averaged = []


    def fit(self, 
            X: np.ndarray, 
            feature_weights: np.ndarray, 
            thr_positive: float, 
            thr_negative:float, 
            threshold: float,
            feature_names: np.ndarray
            ):

        """
        Parameters
        ----------

        X: np.ndarray
         Scaled numpy array to evaluate pipeline

        feature_weights: np.ndarray
            Numpy array with shape X.shape[1] with importance of features

        thr_positive: float
            Threshold for positive class

        thr_negative: float
            Threshold for negative class    

        feature_names: np.ndarray
            Numpy array with shape X.shape[1] with names of features

        """
        

        self.X = X
        self.feature_weights = feature_weights

        self.y = (self.df_kahle_fin['sigma_S_cm'] >= thr_positive).to_numpy()
        self.positive_weights = (1 - self.df_kahle_fin.apply(lambda x: math.erfc((x['sigma_S_cm'] - thr_positive) / x['sigma_S_cm_sem']) / 2, axis=1)).to_numpy()
        self.negative_weights = (self.df_kahle_fin.apply(lambda x: math.erfc((x['sigma_S_cm'] - thr_negative) / x['sigma_S_cm_sem']) / 2, axis=1)).to_numpy()
        self.threshold = threshold
        self.feature_names = feature_names


    def train_loop_with_cv(self, random_split, verbose = False):
        """
        Using leave-one-out method train model_name model

        Parameters
        ----------

        X: np.ndarray
            data
        
        y: np.ndarray
            labels [0, 1]

        weights_for_positive: np.ndarray
            weights for positive class

        weights_for_negative: np.ndarray
            weights for negative class

        model_class: object
            model to evaluate

        verbose: bool
            print train scores

        """
            
        loo = LeaveOneOut()
        loo.get_n_splits(self.X)


        preds_kahle_loop = np.zeros(len(self.y))
        
        if self.X_mpdb is not None:
            preds_mpdb_loop = np.zeros(self.X_mpdb.shape[0])
        else:
            preds_mpdb_loop = None

        if self.X_exp is not None:
            preds_exp_loop = np.zeros(self.X_exp.shape[0])
        else:
            preds_exp_loop = None

        train_roc_auc_loop = np.zeros(len(self.y))

        for i, (train_index, test_index) in enumerate(loo.split(self.X)):

            positive_weights = self.positive_weights[train_index]
            negative_weights = self.negative_weights[train_index]

            X_train = self.X[train_index, :]
            y_train = self.y[train_index]

            X_scaled_positive = X_train[positive_weights > self.threshold]
            X_scaled_negative = X_train[negative_weights > self.threshold]
    
            labels = np.hstack((np.ones(X_scaled_positive.shape[0]), np.zeros(X_scaled_negative.shape[0])))
            weights = np.hstack([positive_weights[positive_weights > self.threshold], negative_weights[negative_weights > self.threshold]])
            X_scaled = np.vstack([X_scaled_positive, X_scaled_negative])

            model = LogisticRegressionCV(**self.params, random_state = random_split)
            model.fit(X_scaled, labels, sample_weight = weights)

            X_test = self.X[test_index]
            preds_kahle_loop[test_index] = model.predict_proba(X_test)[:, 1]
            train_roc_auc_loop[i] = roc_auc_score(labels, model.predict_proba(X_scaled)[:, 1])

            if i == 0 and verbose == True:
                print(f'roc-auc on train for {i} fold with size {X_scaled.shape[0]}: {train_roc_auc_loop[i]}')

            if self.X_mpdb is not None:
                preds_mpdb_loop += model.predict_proba(self.X_mpdb)[:, 1]

            if self.X_exp is not None:
                preds_exp_loop += model.predict_proba(self.X_exp)[:, 1]

        estimated_mean, estimated_std = bootstrap_roc_auc(1000, self.y, preds_kahle_loop)

        if verbose == True:
            print(f'test roc-auc mean: {estimated_mean}, std: {estimated_std}')
            print(f'mean train roc-auc: {np.mean(train_roc_auc_loop)}')
        

        if self.model_name == 'catboost':
            self.feature_importance += model.get_feature_importance()
            self.feature_entarances += model.get_feature_importance() != 0

        if self.model_name == 'logreg':
            self.feature_importance += model.coef_[0]
            self.feature_entarances += model.coef_[0] != 0

        if self.model_name == 'lightgbm':
            self.feature_importance += model.feature_importances_
            self.feature_entarances += model.feature_importances_ != 0


        number_iterations = self.X.shape[0]

        preds_kahle_loop /= number_iterations
        preds_mpdb_loop /= number_iterations
        preds_exp_loop /= number_iterations

        return train_roc_auc_loop, preds_kahle_loop, preds_mpdb_loop, preds_exp_loop



    def train_loop(self, random_split, verbose = False):
        """
        Using leave-one-out method train model_name model

        Parameters
        ----------

        X: np.ndarray
            data
        
        y: np.ndarray
            labels [0, 1]

        weights_for_positive: np.ndarray
            weights for positive class

        weights_for_negative: np.ndarray
            weights for negative class

        model_class: object
            model to evaluate

        verbose: bool
            print train scores

        """
            
        loo = LeaveOneOut()
        loo.get_n_splits(self.X)


        preds_kahle_loop = np.zeros(len(self.y))
        
        if self.X_mpdb is not None:
            preds_mpdb_loop = np.zeros(self.X_mpdb.shape[0])
        else:
            preds_mpdb_loop = None

        if self.X_exp is not None:
            preds_exp_loop = np.zeros(self.X_exp.shape[0])
        else:
            preds_exp_loop = None

        train_roc_auc_loop = np.zeros(len(self.y))

        for i, (train_index, test_index) in enumerate(loo.split(self.X)):

            positive_weights = self.positive_weights[train_index]
            negative_weights = self.negative_weights[train_index]

            X_train = self.X[train_index, :]
            y_train = self.y[train_index]

            X_scaled_positive = X_train[positive_weights > self.threshold]
            X_scaled_negative = X_train[negative_weights > self.threshold]
    
            labels = np.hstack((np.ones(X_scaled_positive.shape[0]), np.zeros(X_scaled_negative.shape[0])))
            weights = np.hstack([positive_weights[positive_weights > self.threshold], negative_weights[negative_weights > self.threshold]])
            X_scaled = np.vstack([X_scaled_positive, X_scaled_negative])
            

            if self.model_name == 'lightgbm':
                model = LGBMClassifier(**self.params, seed = random_split, bagging_seed = random_split, feature_weights = self.feature_weights)
        
            if self.model_name == 'catboost':
                model = CatBoostClassifier(**self.params, eval_metric='AUC', verbose = False, random_state = random_split, feature_weights = self.feature_weights)

            if self.model_name == 'logreg':
                model = LogisticRegression(**self.params, verbose = False, random_state = random_split)

            if self.model_name == 'random':
                model = RandomModel(random_state = random_split)


            if isinstance(model, CatBoostClassifier):
                train_pool = Pool(X_scaled, label = labels, weight = weights)
                model.fit(train_pool)

            else:
                model.fit(X_scaled, labels, sample_weight = weights)

            X_test = self.X[test_index]
            preds_kahle_loop[test_index] = model.predict_proba(X_test)[:, 1]
            train_roc_auc_loop[i] = roc_auc_score(labels, model.predict_proba(X_scaled)[:, 1])

            if i == 0 and verbose == True:
                print(f'roc-auc on train for {i} fold with size {X_scaled.shape[0]}: {train_roc_auc_loop[i]}')

            if self.X_mpdb is not None:
                preds_mpdb_loop += model.predict_proba(self.X_mpdb)[:, 1]

            if self.X_exp is not None:
                preds_exp_loop += model.predict_proba(self.X_exp)[:, 1]

        estimated_mean, estimated_std = bootstrap_roc_auc(1000, self.y, preds_kahle_loop)

        if verbose == True:
            print(f'test roc-auc mean: {estimated_mean}, std: {estimated_std}')
            print(f'mean train roc-auc: {np.mean(train_roc_auc_loop)}')
        

        if self.model_name == 'catboost':
            self.feature_importance += model.get_feature_importance()
            self.feature_entarances += model.get_feature_importance() != 0

        if self.model_name == 'logreg':
            self.feature_importance += model.coef_[0]
            self.feature_entarances += model.coef_[0] != 0

        if self.model_name == 'lightgbm':
            self.feature_importance += model.feature_importances_
            self.feature_entarances += model.feature_importances_ != 0


        number_iterations = self.X.shape[0]

        preds_kahle_loop /= number_iterations
        preds_mpdb_loop /= number_iterations
        preds_exp_loop /= number_iterations

        return train_roc_auc_loop, preds_kahle_loop, preds_mpdb_loop, preds_exp_loop




    def evaluate_with_cv(self, num_of_evaluations = 40, X_mpdb = None, X_exp = None):

        """
        Parameters
        ----------

        num_of_evaluations: int
            Number of iterations to test pipeline

        X_mpdb: np.ndarray
            Numpy array of material project database to make aggregated predictions

        """
        
        self.num_of_evaluations = num_of_evaluations
        self.train_roc_auc = []
        self.test_roc_auc = []
        self.roc_like_comparison = []

        self.feature_importance = np.zeros(self.X.shape[1])
        self.feature_entarances = np.zeros(self.X.shape[1])

        self.random_splits = np.random.randint(10000, size = num_of_evaluations)

        self.preds_kahle = []
        self.preds_mpdb = []
        self.preds_exp = []

        self.X_mpdb = X_mpdb
        self.X_exp = X_exp

        for i in trange(num_of_evaluations):

            if i == 0:
                train_roc_auc_loop, preds_kahle_loop, preds_mpdb_loop, preds_exp_loop = self.train_loop_with_cv(random_split = self.random_splits[i], verbose = True)
            else:
                train_roc_auc_loop, preds_kahle_loop, preds_mpdb_loop, preds_exp_loop = self.train_loop_with_cv(random_split = self.random_splits[i], verbose = False)

            mean_test, std_test = bootstrap_roc_auc(1000, self.y, preds_kahle_loop)

            self.test_roc_auc.append([mean_test, std_test])
            self.train_roc_auc.append([np.mean(train_roc_auc_loop), np.std(train_roc_auc_loop)])

            self.roc_like_comparison.append(calculate_ROClikeComparisonMetricsKahle(self.preds_Kahle2020, preds_kahle_loop))
            self.preds_kahle.append(preds_kahle_loop)
            self.preds_mpdb.append(preds_mpdb_loop)
            self.preds_exp.append(preds_exp_loop)



    def evaluate(self, num_of_evaluations = 40, X_mpdb = None, X_exp = None):

        """
        Parameters
        ----------

        num_of_evaluations: int
            Number of iterations to test pipeline

        X_mpdb: np.ndarray
            Numpy array of material project database to make aggregated predictions

        """
        
        self.num_of_evaluations = num_of_evaluations
        self.train_roc_auc = []
        self.test_roc_auc = []
        self.roc_like_comparison = []

        self.feature_importance = np.zeros(self.X.shape[1])
        self.feature_entarances = np.zeros(self.X.shape[1])

        self.random_splits = np.random.randint(10000, size = num_of_evaluations)

        self.preds_kahle = []
        self.preds_mpdb = []
        self.preds_exp = []

        self.X_mpdb = X_mpdb
        self.X_exp = X_exp

        for i in trange(num_of_evaluations):

            if i == 0:
                train_roc_auc_loop, preds_kahle_loop, preds_mpdb_loop, preds_exp_loop = self.train_loop(random_split = self.random_splits[i], verbose = True)
            else:
                train_roc_auc_loop, preds_kahle_loop, preds_mpdb_loop, preds_exp_loop = self.train_loop(random_split = self.random_splits[i], verbose = False)

            mean_test, std_test = bootstrap_roc_auc(1000, self.y, preds_kahle_loop)

            self.test_roc_auc.append([mean_test, std_test])
            self.train_roc_auc.append([np.mean(train_roc_auc_loop), np.std(train_roc_auc_loop)])

            self.roc_like_comparison.append(calculate_ROClikeComparisonMetricsKahle(self.preds_Kahle2020, preds_kahle_loop))
            self.preds_kahle.append(preds_kahle_loop)
            self.preds_mpdb.append(preds_mpdb_loop)
            self.preds_exp.append(preds_exp_loop)


    def save_mpdb_preds(self, path):

        """
        Parameters
        ----------

        path: str
            Path to save mpdb predictions

        """
        np.savetxt(path, self.preds_mpdb)


    def save_exp_preds(self, path):

        """
        Parameters
        ----------

        path: str
            Path to save exp predictions

        """
        np.savetxt(path, self.preds_exp)


    def get_aggregated_statistics(self):

        preds_kahle_agg = np.zeros(self.preds_kahle[0].shape)
        preds_mpdb_agg = np.zeros(self.preds_mpdb[0].shape)
        preds_exp_agg =np.zeros(self.preds_exp[0].shape)

        self.roc_auc_averaged = []
        self.roc_like_comparison_averaged = []

        self.preds_mpdb_averaged = []
        self.preds_kahle_averaged = []
        self.preds_exp_averaged = []

        for index in range(self.num_of_evaluations):

            preds_kahle_agg += self.preds_kahle[index]
            preds_mpdb_agg += self.preds_mpdb[index]
            preds_exp_agg += self.preds_exp[index]

            estimated_mean, estimated_std = bootstrap_roc_auc(1000, self.y, preds_kahle_agg / (index + 1))
            self.roc_auc_averaged.append([estimated_mean, estimated_std])
            self.roc_like_comparison_averaged.append(calculate_ROClikeComparisonMetricsKahle(self.preds_Kahle2020, preds_kahle_agg / (index + 1)))

            self.preds_kahle_averaged.append(preds_kahle_agg / (index + 1))
            self.preds_mpdb_averaged.append(preds_mpdb_agg / (index + 1))
            self.preds_exp_averaged.append(preds_exp_agg / (index + 1))

        self.roc_auc_averaged = np.array(self.roc_auc_averaged)
        self.roc_like_comparison_averaged = np.array(self.roc_like_comparison_averaged)

    def plot_statistics(self):

        """
        Plot results of model evaluation
        """

        if len(self.roc_auc_averaged) == 0:
            self.get_aggregated_statistics()
            
        fig, axs = plt.subplots(4, 2, figsize=(30, 25))
        fig.suptitle(f'', fontsize=15)

        axs[0, 0].set_title('roc-auc on train')
        axs[0, 0].errorbar(np.arange(len(self.train_roc_auc)), np.array(self.train_roc_auc)[:, 0], yerr = np.array(self.train_roc_auc)[:, 1], capsize=3, fmt="r--o", ecolor = "black");
        axs[0, 0].set_ylabel('roc-auc')
        axs[0, 0].set_xlabel('index of model')

        axs[0, 1].set_title('difference between previous aggregation and current')

        lenn = [10, 100, 500]
        evaluations = np.arange(1, self.num_of_evaluations)
        for length in lenn:
            diff = []
            for i in evaluations:
                prev = set(np.argsort(np.array(self.preds_mpdb_averaged[i - 1]))[-length:])
                cur = set(np.argsort(np.array(self.preds_mpdb_averaged[i]))[-length:])
                diff.append(length - len(prev & cur))
            axs[0, 1].plot(evaluations, diff, label= f'size of top: {length}, last diff: {diff[-1]}')

        axs[0, 1].legend()
        axs[0, 1].set_ylabel('difference')
        axs[0, 1].set_xlabel('index of model')


        axs[1, 0].set_title('roc-auc on test')
        axs[1, 0].errorbar(np.arange(len(self.test_roc_auc)), np.array(self.test_roc_auc)[:, 0], yerr = np.array(self.test_roc_auc)[:, 1], capsize=3, fmt="r--o", ecolor = "black");
        axs[1, 0].set_ylabel('roc-auc')
        axs[1, 0].set_xlabel('index of model')

        axs[1, 1].set_title('roc-auc averaged by prev preds on test')
        axs[1, 1].errorbar(np.arange(len(self.roc_auc_averaged)), self.roc_auc_averaged[:, 0], yerr = self.roc_auc_averaged[:, 1], capsize=3, fmt="r--o", ecolor = "black");
        axs[1, 1].set_ylabel('roc-auc')
        axs[1, 1].set_xlabel('number of models')

        axs[2, 0].set_title('roc_like_comparison')
        axs[2, 0].plot(np.arange(len(self.test_roc_auc)), self.roc_like_comparison);
        axs[2, 0].set_ylabel('metric value')
        axs[2, 0].set_xlabel('index of model');

        axs[2, 1].set_title('roc_like_comparison averaged')
        axs[2, 1].plot(np.arange(len(self.test_roc_auc)), self.roc_like_comparison_averaged);
        axs[2, 1].set_ylabel('metric value')
        axs[2, 1].set_xlabel('number of models');

        if self.model_name == 'logreg':
            self.feature_importance = np.abs(self.feature_importance)

        idx = np.argsort(self.feature_importance)[-50:]

        axs[3, 0].xaxis.set_tick_params(rotation=90)
        axs[3, 0].set_title('feature importance top-50')
        axs[3, 0].set_ylabel('importance')
        axs[3, 0].set_xlabel('feature names');
        axs[3, 0].bar(self.feature_names[idx], self.feature_importance[idx] / self.num_of_evaluations)


        idx = np.argsort(self.feature_entarances)[-50:]

        axs[3, 1].xaxis.set_tick_params(rotation=90)
        axs[3, 1].set_title('feature entrances top-50')
        axs[3, 1].set_ylabel('number')
        axs[3, 1].set_xlabel('feature names');
        axs[3, 1].bar(self.feature_names[idx], self.feature_entarances[idx]);


    def plot_feature_importance(self):

        fig, axs = plt.subplots(1, 1, figsize=(15, 10))
        fig.suptitle(f'', fontsize=15)

        if self.model_name == 'logreg':
            self.feature_importance = np.abs(self.feature_importance)

        idx = np.argsort(self.feature_importance)[-50:]

        axs.xaxis.set_tick_params(rotation=90)
        axs.set_title('feature importance top-50')
        axs.set_ylabel('importance')
        axs.set_xlabel('feature names');
        axs.bar(self.feature_names[idx], self.feature_importance[idx] / self.num_of_evaluations)


    def plot_distribution_compared(self, title):

        """
        Plot results of model evaluation
        """

        """
        Parameters
        ----------

        title: str
            Title of plot

        """    
        plot_distribution_compared(
        self.df_kahle_fin,
        self.preds_kahle[0],
        np.array(self.preds_kahle).sum(axis = 0) / len(self.preds_kahle), 
        title
        )


    def show_results(self):

        """
        Show results of model evaluation as Pandas DataFrame
        """


        if len(self.roc_auc_averaged) == 0:
            self.get_aggregated_statistics()

        results = pd.DataFrame({
              'mean train roc-auc averaged by 40 models': [np.array(self.train_roc_auc)[:, 0].mean()],
              'roc_auc on test by aggregated preds': [self.roc_auc_averaged[-1][0]],
              'mean test roc-auc averaged by 40 models': [np.array(self.test_roc_auc)[:, 0].mean()],
              'roc_like_comparison by aggregated preds': [self.roc_like_comparison_averaged[-1]],
              'roc_like_comparison averaged by 40 models': [np.array(self.roc_like_comparison).mean()],
              'roc_like_comparison std': [np.array(self.roc_like_comparison).std()]
              }).T
              
    
        results = results.reset_index()
        results.columns = ['statistic', 'result']

        return results
