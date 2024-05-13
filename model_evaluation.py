import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from utils import bootstrap_roc_auc

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from training_functions import train_loop, calculate_ROClikeComparisonMetrics, plot_distribution_compared
from tqdm import trange

class ModelEvaluation():



    """
    This class is designed to test the models and pipelines
    """



    def __init__(self, 
                 df_kahle_fin: pd.DataFrame, 
                 model_name: str, 
                 params: dict):

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
        self.params = params
        self.roc_auc_averaged = []
        self.roc_like_comparison_averaged = []

    def fit(self, 
            X: np.ndarray, 
            feature_weights: np.ndarray, 
            thr_positive: float, 
            thr_negative:float, 
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

        self.feature_names = feature_names

    def evaluate(self, num_of_evaluations = 40, X_mpdb = None):

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


        for i in trange(num_of_evaluations):

            if self.model_name == 'lightgbm':
                model_class = LGBMClassifier(**self.params, seed = self.random_splits[i], bagging_seed = self.random_splits[i], feature_weights = self.feature_weights)
        
            if self.model_name == 'catboost':
                model_class = CatBoostClassifier(**self.params, eval_metric='AUC', verbose = False, random_state = self.random_splits[i], feature_weights = self.feature_weights)

            if self.model_name == 'logreg':
                model_class = LogisticRegression(**self.params, verbose = False, random_state = self.random_splits[i])

            if i == 0:
                preds_kahle_loop, model, mean_test, std_test, mean_train, std_train, preds_mpdb_loop = train_loop(self.X, self.y, self.positive_weights, self.negative_weights, model_class, X_mpdb = X_mpdb)
            else:
                preds_kahle_loop, model, mean_test, std_test, mean_train, std_train, preds_mpdb_loop = train_loop(self.X, self.y, self.positive_weights, self.negative_weights, model_class, verbose = False, X_mpdb = X_mpdb)

            self.test_roc_auc.append([mean_test, std_test])
            self.train_roc_auc.append([mean_train, std_train])
            self.roc_like_comparison.append(calculate_ROClikeComparisonMetrics(self.df_kahle_fin, preds_kahle_loop)['score']['preds'])

            self.preds_kahle.append(preds_kahle_loop)
            if X_mpdb is not None:
                self.preds_mpdb.append(preds_mpdb_loop)

            if self.model_name == 'catboost':
                self.feature_importance += model.get_feature_importance()
                self.feature_entarances += model.get_feature_importance() != 0

            if self.model_name == 'logreg':
                self.feature_importance += model.coef_[0]
                self.feature_entarances += model.coef_[0] != 0

            if self.model_name == 'lightgbm':
                self.feature_importance += model.feature_importances_
                self.feature_entarances += model.feature_importances_ != 0

    def save_mpdb_preds(self, path):

        """
        Parameters
        ----------

        path: str
            Path to save mpdb predictions

        """

        np.savetxt(path, self.preds_mpdb)

    def get_aggregated_statistics(self):

        preds_agg = np.zeros(self.preds_kahle[0].shape)
        preds_mpdb_agg = np.zeros(self.preds_mpdb[0].shape)
        self.roc_auc_averaged = []
        self.roc_like_comparison_averaged = []
        self.preds_mpdb_averaged = []

        for index in range(self.num_of_evaluations):

            preds_agg += self.preds_kahle[index]
            preds_mpdb_agg += self.preds_mpdb[index]

            estimated_mean, estimated_std = bootstrap_roc_auc(1000, self.y, preds_agg / (index + 1))
            self.roc_auc_averaged.append([estimated_mean, estimated_std])

            self.roc_like_comparison_averaged.append(calculate_ROClikeComparisonMetrics(self.df_kahle_fin, preds_agg / (index + 1))['score']['preds'])
            self.preds_mpdb_averaged.append(preds_mpdb_agg / (index + 1))
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
        axs[3, 0].bar(self.feature_names[idx], self.feature_importance[idx])


        idx = np.argsort(self.feature_entarances)[-50:]

        axs[3, 1].xaxis.set_tick_params(rotation=90)
        axs[3, 1].set_title('feature entrances top-50')
        axs[3, 1].set_ylabel('number')
        axs[3, 1].set_xlabel('feature names');
        axs[3, 1].bar(self.feature_names[idx], self.feature_entarances[idx]);


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
