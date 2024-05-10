
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.preprocessing import StandardScaler
from scoring import bootstrap_roc_auc
from data_preprocessing import *
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

def plot_distribution(
        df: pd.DataFrame, 
        preds: np.ndarray, 
        title: str
        ):

    """
    Plot distribution of model prediction 

    Parameters
    ----------

    df: pd.DataFrame
        Pandas Dataframe with columns:  diffusion_mean_cm2_s [cm^2/s],
                                        sigma_S_cm [S/cm]
                                        group [A, B1, B2, C]
    
    preds: np.ndarray
        Numpy array of model preds in a range [0; 1]

    title: str
        Title of the plot

    """

    
    df['preds'] = preds

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(title, fontsize=15)


    A = df[df['group'] == 'A']
    B1 = df[df['group'] == 'B1']
    B2 = df[df['group'] == 'B2']
    C = df[df['group'] == 'C']


    axs[0].scatter(np.log10(A['diffusion_mean_cm2_s']).to_list(), A['preds'].to_list(), label='A', c = 'pink')
    axs[0].scatter(np.log10(B1['diffusion_mean_cm2_s']).to_list(), B1['preds'].to_list(), label='B1', c = 'purple')
    axs[0].scatter(np.log10(B2['diffusion_mean_cm2_s']).to_list(), B2['preds'].to_list(), label='B2', c = 'lightblue')
    axs[0].scatter(np.log10(C['diffusion_mean_cm2_s']).to_list(), C['preds'].to_list(), label='C', c = 'lightgreen')

    axs[0].set_ylabel('probability')
    axs[0].set_xlabel('diffusion coefficient, $\log_{10}D[\\frac{cm^2}{s}]$')

    axs[1].scatter(np.log10(A['sigma_S_cm']).to_list(), A['preds'].to_list(), label='A', c = 'pink')
    axs[1].scatter(np.log10(B1['sigma_S_cm']).to_list(), B1['preds'].to_list(), label='B1', c = 'purple')
    axs[1].scatter(np.log10(B2['sigma_S_cm']).to_list(), B2['preds'].to_list(), label='B2', c = 'lightblue')
    axs[1].scatter(np.log10(C['sigma_S_cm']).to_list(), C['preds'].to_list(), label='C', c = 'lightgreen')

    axs[1].set_ylabel('probability')
    axs[1].set_xlabel('sigma_S_cm')

    plt.legend();




def train_loop(
        X: np.ndarray, 
        y: np.ndarray, 
        weights_for_positive: np.ndarray,
        weights_for_negative: np.ndarray,
        model_class: object, 
        threshold: float = 1e-2, 
        verbose: bool = True,
        X_mpdb: np.ndarray = None,
        ):

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
    loo.get_n_splits(X)
    preds = np.zeros(len(y))

    preds_mpdb = np.zeros(X_mpdb.shape[0])
        
    train_roc_auc = np.zeros(len(y))

    scaler = StandardScaler().fit(X_mpdb)

    for i, (train_index, test_index) in enumerate(tqdm(loo.split(X), total=len(y))):
        positive_weights = weights_for_positive[train_index]
        negative_weights = weights_for_negative[train_index]

        X_train = X[train_index, :]
        y_train = y[train_index]

        X_scaled_positive = scaler.transform(X_train[positive_weights > threshold])
        X_scaled_negative = scaler.transform(X_train[negative_weights > threshold])
 
        labels = np.hstack((np.ones(X_scaled_positive.shape[0]), np.zeros(X_scaled_negative.shape[0])))
        weights = np.hstack([positive_weights[positive_weights > threshold], negative_weights[negative_weights > threshold]])
        X_scaled = np.vstack([X_scaled_positive, X_scaled_negative])
        model = model_class

        if isinstance(model, CatBoostClassifier):
            train_pool = Pool(X_scaled, label = labels, weight = weights)
            model.fit(train_pool)

            X_test = scaler.transform(X[test_index])
            preds[test_index] = model.predict_proba(X_test)[:, 1]

            train_roc_auc[i] = roc_auc_score(labels, model.predict_proba(X_scaled)[:, 1])

        else:
            model.fit(X_scaled, labels, sample_weight = weights)
            X_test = scaler.transform(X[test_index])
            preds[test_index] = model.predict_proba(X_test)[:, 1]
            train_roc_auc[i] = roc_auc_score(labels, model.predict_proba(X_scaled)[:, 1])


        if i == 0 and verbose == True:
            print(f'roc-auc on train for {i} fold with size {X_scaled.shape[0]}: {train_roc_auc[i]}')

        preds_mpdb += model.predict_proba(scaler.transform(X_mpdb))[:, 1]

    estimated_mean, estimated_std = bootstrap_roc_auc(1000, y, preds)

    if verbose == True:
        print(f'test roc-auc mean: {estimated_mean}, std: {estimated_std}')
        print(f'mean train roc-auc: {np.mean(train_roc_auc)}')
    
    return preds, model, estimated_mean, estimated_std, np.mean(train_roc_auc), np.std(train_roc_auc), preds_mpdb


def train_simple_loop(
        X: np.ndarray, 
        y: np.ndarray, 
        model_class: object, 
        verbose: bool = True):

    """
    Train model

    Parameters
    ----------

    X: np.ndarray
        data
    y: np.ndarray
        labels [0, 1]
    model_class: object
        model to evaluate
    verbose: bool
        print train scores

    """

    loo = LeaveOneOut()
    loo.get_n_splits(X)
    preds = np.zeros(len(y))
    train_roc_auc = np.zeros(len(y))
    for i, (train_index, test_index) in enumerate(tqdm(loo.split(X), total=len(y))):

        scaler = StandardScaler().fit(X[train_index, :])

        X_train = X[train_index, :]
        y_train = y[train_index]
        X_scaled = scaler.transform(X_train)

        model = model_class
        model_class.fit(X_scaled, y_train)

        X_test = scaler.transform(X[test_index])

        preds[test_index] = model.predict_proba(X_test)[:, 1]

        preds_train = model.predict(X_scaled)
        train_roc_auc[i] = roc_auc_score(y_train, preds_train)
        
    estimated_mean, estimated_std = bootstrap_roc_auc(1000, y, preds)

    if verbose == True:
        print(estimated_mean, estimated_std)
        print(f'mean train roc-auc {train_roc_auc.mean()}')

    return preds, model


def feature_importance_hist(
        feature_importance: np.ndarray, 
        name: str):

    """
    Plot feature importance histogram

    Parameters
    ----------
    feature_importance: np.ndarray
        model feature importance array
    name: str
        title name

    """
        
    bins = np.linspace(-1, 1, 100)
    plt.figure(figsize=(15, 10))
    plt.title(name)
    plt.hist(feature_importance, bins, alpha=0.5)
    plt.hist(feature_importance[(feature_importance >= -1e-2) & (feature_importance <= 1e-2)], bins, alpha=0.5, label='features less then 1e-2')
    plt.legend(loc='upper right')
    plt.show()
    
def plot_feature_importance(
        feature_importance: np.ndarray, 
        feature_names: np.ndarray):

    """
    Plot feature importance histogram

    Parameters
    ----------
    feature_importance: np.ndarray
        model feature importance array
    feature_names: str
        feature names array

    """
        
    idx = np.argsort(feature_importance)[-50:]
    plt.figure(figsize=(20, 10))
    plt.xticks(rotation=90)
    plt.title('feature importance top-50')
    plt.bar(feature_names[idx], feature_importance[idx])
    plt.show()



from misc_utils.feature_analysis import *

def calculate_ROClikeComparisonMetrics(
                            df: pd.DataFrame, 
                            preds: np.ndarray):
    
    """
    Plot feature importance histogram

    Parameters
    ----------
    df: pd.DataFrame
        Pandas dataframe with ['sigma_S_cm', 'sigma_S_cm_err']
    preds: np.ndarray
        predictions of model

    """

    rlcm = ROClikeComparisonMetrics()
    df['preds'] = preds
    features_meta_info = pd.DataFrame({'name': ['preds'], 'weighted_direction': [1]})
    features_meta_info = features_meta_info.set_index(features_meta_info.columns[0])

    feature_scores = rlcm.eval_features(preds_df = df[['sigma_S_cm', 'sigma_S_cm_err', 'preds']], features_meta_info = features_meta_info)

    return feature_scores


def calculate_statistics(df_kahle_fin, all_preds, y):
    preds_agg = np.zeros(all_preds[0].shape)
    roc_auc_averaged = []
    roc_like_comparison_averaged = []
    roc_like_comparison = []

    for index, pred in enumerate(all_preds):
        preds_agg += pred

        estimated_mean, estimated_std = bootstrap_roc_auc(1000, y, preds_agg / (index + 1))
        roc_auc_averaged.append([estimated_mean, estimated_std])

        roc_like_comparison_averaged.append(calculate_ROClikeComparisonMetrics(df_kahle_fin, preds_agg / (index + 1))['score']['preds'])
        roc_like_comparison.append()
    roc_auc_averaged = np.array(roc_auc_averaged)
    roc_like_comparison_averaged = np.array(roc_like_comparison_averaged)


def plot_statistics(df_kahle_fin, feature_names, feature_entarances, feature_importance, train_roc_auc, test_roc_auc, roc_like_comparison, all_preds, y):

    preds_agg = np.zeros(all_preds[0].shape)
    roc_auc_averaged = []
    roc_like_comparison_averaged = []

    for index, pred in enumerate(all_preds):
        preds_agg += pred

        estimated_mean, estimated_std = bootstrap_roc_auc(1000, y, preds_agg / (index + 1))
        roc_auc_averaged.append([estimated_mean, estimated_std])

        roc_like_comparison_averaged.append(calculate_ROClikeComparisonMetrics(df_kahle_fin, preds_agg / (index + 1))['score']['preds'])

    roc_auc_averaged = np.array(roc_auc_averaged)
    roc_like_comparison_averaged = np.array(roc_like_comparison_averaged)


    fig, axs = plt.subplots(4, 2, figsize=(30, 25))
    fig.suptitle(f'', fontsize=15)

    axs[0, 0].set_title('roc-auc on train')
    axs[0, 0].errorbar(np.arange(len(train_roc_auc)), np.array(train_roc_auc)[:, 0], yerr = np.array(train_roc_auc)[:, 1], capsize=3, fmt="r--o", ecolor = "black");
    axs[0, 0].set_ylabel('roc-auc')
    axs[0, 0].set_xlabel('index of model')


    axs[1, 0].set_title('roc-auc on test')
    axs[1, 0].errorbar(np.arange(len(test_roc_auc)), np.array(test_roc_auc)[:, 0], yerr = np.array(test_roc_auc)[:, 1], capsize=3, fmt="r--o", ecolor = "black");
    axs[1, 0].set_ylabel('roc-auc')
    axs[1, 0].set_xlabel('index of model')

    axs[1, 1].set_title('roc-auc averaged by prev preds on test')
    axs[1, 1].errorbar(np.arange(len(roc_auc_averaged)), roc_auc_averaged[:, 0], yerr = roc_auc_averaged[:, 1], capsize=3, fmt="r--o", ecolor = "black");
    axs[1, 1].set_ylabel('roc-auc')
    axs[1, 1].set_xlabel('number of models')

    axs[2, 0].set_title('roc_like_comparison')
    axs[2, 0].plot(np.arange(len(test_roc_auc)), roc_like_comparison);
    axs[2, 0].set_ylabel('metric value')
    axs[2, 0].set_xlabel('index of model');

    axs[2, 1].set_title('roc_like_comparison averaged')
    axs[2, 1].plot(np.arange(len(test_roc_auc)), roc_like_comparison_averaged);
    axs[2, 1].set_ylabel('metric value')
    axs[2, 1].set_xlabel('number of models');


    idx = np.argsort(feature_importance)[-50:]

    axs[3, 0].xaxis.set_tick_params(rotation=90)
    axs[3, 0].set_title('feature importance top-50')
    axs[3, 0].set_ylabel('importance')
    axs[3, 0].set_xlabel('feature names');
    axs[3, 0].bar(feature_names[idx], feature_importance[idx])


    idx = np.argsort(feature_entarances)[-50:]

    axs[3, 1].xaxis.set_tick_params(rotation=90)
    axs[3, 1].set_title('feature entrances top-50')
    axs[3, 1].set_ylabel('number')
    axs[3, 1].set_xlabel('feature names');
    axs[3, 1].bar(feature_names[idx], feature_entarances[idx]);

    return roc_like_comparison_averaged, roc_auc_averaged


from pymatgen.core import Structure
from ase import units



def plot_diffusion_distribution(df, axs, preds, i, j, title):

    df['preds'] = preds

    A = df[df['group'] == 'A']
    B1 = df[df['group'] == 'B1']
    B2 = df[df['group'] == 'B2']
    C = df[df['group'] == 'C']


    axs[i, j].scatter(np.log10(A['diffusion_mean_cm2_s']).to_list(), A['preds'].to_list(), label='A', c = 'pink')
    axs[i, j].scatter(np.log10(B1['diffusion_mean_cm2_s']).to_list(), B1['preds'].to_list(), label='B1', c = 'purple')
    axs[i, j].scatter(np.log10(B2['diffusion_mean_cm2_s']).to_list(), B2['preds'].to_list(), label='B2', c = 'lightblue')
    axs[i, j].scatter(np.log10(C['diffusion_mean_cm2_s']).to_list(), C['preds'].to_list(), label='C', c = 'lightgreen')

    axs[i, j].set_ylabel('probability')
    axs[i, j].set_xlabel('diffusion coefficient, $\log_{10}D[\\frac{cm^2}{s}]$')

    axs[i, j].set_title(title)



def plot_axs_hist(axs, weight_for_positive, weight_for_negative, title):

    bins = np.linspace(0, 2, 15)
    
    axs.hist(weight_for_positive, alpha=0.5, label='positive weight', bins = bins, width = 0.02)
    axs.hist(weight_for_negative, alpha=0.5, label='negative weight', bins = bins, width = 0.02)
    axs.set_xlabel('weight')
    axs.set_ylabel('count')
    axs.set_title(title)
    axs.legend(loc='upper right')


def plot_weight_distribution(df, thr_positive, thr_negative):

    weight_for_positive = 1 - df.apply(lambda x: math.erfc((x['sigma_S_cm'] - thr_positive) / x['sigma_S_cm_sem']) / 2, axis=1)
    weight_for_negative = df.apply(lambda x: math.erfc((x['sigma_S_cm'] - thr_negative) / x['sigma_S_cm_sem']) / 2, axis=1)

    A = df[df['group'] == 'A'].index
    B1 = df[df['group'] == 'B1'].index
    B2 = df[df['group'] == 'B2'].index
    C = df[df['group'] == 'C'].index

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'weights for positive and negative class with positive threshold {thr_positive}, negative threshold {thr_negative}', fontsize=15)

    plot_axs_hist(axs[0][0], weight_for_positive[A], weight_for_negative[A], 'group A')
    plot_axs_hist(axs[1][0], weight_for_positive[B1], weight_for_negative[B1], 'group B1')
    plot_axs_hist(axs[0][1], weight_for_positive[B2], weight_for_negative[B2], 'group B2')
    plot_axs_hist(axs[1][1], weight_for_positive[C], weight_for_negative[C], 'group C')


def plot_ax(axs, y, x, indexes):

    labels = ['A', 'B1', 'B2', 'C']
    colors = ['pink', 'purple', 'lightblue', 'lightgreen']

    for index, label, color in zip(indexes, labels, colors):
        axs.scatter(y[index], x[index], label=label, c = color)

def plot_distribution_compared(
        df: pd.DataFrame, 
        single_preds: np.ndarray, 
        aggregated_preds: np.ndarray, 
        title: str
        ):

    """
    Plot distribution of model prediction 

    Parameters
    ----------

    df: pd.DataFrame
        Pandas Dataframe with columns:  diffusion_mean_cm2_s [cm^2/s],
                                        sigma_S_cm [S/cm]
                                        group [A, B1, B2, C]
    
    preds: np.ndarray
        Numpy array of model preds in a range [0; 1]

    title: str
        Title of the plot

    """

    
    df['preds'] = single_preds

    A = df[df['group'] == 'A'].index
    B1 = df[df['group'] == 'B1'].index
    B2 = df[df['group'] == 'B2'].index
    C = df[df['group'] == 'C'].index

    indexes = [A, B1, B2, C]

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(title, fontsize=12)

    plot_ax(axs[0, 0], np.log10(df['diffusion_mean_cm2_s']).to_numpy(), single_preds, indexes)
    axs[0, 0].set_ylabel('probability')
    axs[0, 0].set_xlabel('diffusion coefficient, $\log_{10}D[\\frac{cm^2}{s}]$')
    axs[0, 0].set_title('distribution of np.log10(diffusivity) for single preds')


    plot_ax(axs[1, 0], df['sigma_S_cm'].to_numpy(), single_preds, indexes)
    axs[1, 0].set_ylabel('probability')
    axs[1, 0].set_xlabel('electrolytic conductivity, $\sigma[\\frac{\S}{cm}]$')
    axs[1, 0].set_title('distribution of conductivity for single preds')


    plot_ax(axs[0, 1], np.log10(df['diffusion_mean_cm2_s']).to_numpy(), aggregated_preds, indexes)
    axs[0, 1].set_ylabel('probability')
    axs[0, 1].set_xlabel('diffusion coefficient, $\log_{10}D[\\frac{cm^2}{s}]$')
    axs[0, 1].set_title('distribution of np.log10(diffusivity) for aggregated preds')

    plot_ax(axs[1, 1], df['sigma_S_cm'].to_numpy(), aggregated_preds, indexes)
    axs[1, 1].set_ylabel('probability')
    axs[1, 1].set_xlabel('electrolytic conductivity, $\sigma[\\frac{\S}{cm}]$')
    axs[1, 1].set_title('distribution of conductivity for aggregated preds')

    plt.legend();


def evaluate_parameters(params, X, df_kahle_fin, feature_weights, thr_positive = 1e-1, thr_negative = 1e-2, model_name = 'catboost', X_mpdb = None):

    y = (df_kahle_fin['sigma_S_cm'] >= thr_positive).to_numpy()
    positive_weights = (1 - df_kahle_fin.apply(lambda x: math.erfc((x['sigma_S_cm'] - thr_positive) / x['sigma_S_cm_sem']) / 2, axis=1)).to_numpy()
    negative_weights = (df_kahle_fin.apply(lambda x: math.erfc((x['sigma_S_cm'] - thr_negative) / x['sigma_S_cm_sem']) / 2, axis=1)).to_numpy()


    train_roc_auc = []
    test_roc_auc = []
    roc_like_comparison = []
    feature_importance = np.zeros(X.shape[1])
    random_splits = []

    preds_kahle = []
    preds_mpdb = []



    
    feature_entarances = np.zeros(X.shape[1])
    for rs in np.random.randint(1000, size = 40):

        if model_name == 'lightgbm':
            model_class = LGBMClassifier(**params, seed = rs, bagging_seed = rs, feature_weights = feature_weights)
        
        if model_name == 'catboost':
            model_class = CatBoostClassifier(**params, eval_metric='AUC', verbose = False, random_state = rs, feature_weights = feature_weights)

        preds_kahle_loop, model, mean_test, std_test, mean_train, std_train, preds_mpdb_loop = train_loop(X, y, positive_weights, negative_weights, model_class, X_mpdb = X_mpdb)

        test_roc_auc.append([mean_test, std_test])
        train_roc_auc.append([mean_train, std_train])
        roc_like_comparison.append(calculate_ROClikeComparisonMetrics(df_kahle_fin, preds_kahle_loop)['score']['preds'])
        random_splits.append(rs)

        preds_kahle.append(preds_kahle_loop)
        if X_mpdb is not None:
            preds_mpdb.append(preds_mpdb_loop / len(preds_kahle_loop))

        if model_name == 'catboost':
            feature_importance += model.get_feature_importance()
            feature_entarances += model.get_feature_importance() != 0
        else:
            feature_importance += model.feature_importances_
            feature_entarances += model.feature_importances_ != 0

    return test_roc_auc, train_roc_auc, roc_like_comparison, preds_kahle, feature_importance, feature_entarances, y, preds_mpdb



featurizers_mapping = {
    'ape_features': 'AtomicPackingEfficiency',
    'bc_features': 'BandCenter',
    'bf_features': 'BondFractions',
    'co_features': 'ChemicalOrdering',
    'density_features': 'DensityFeatures',
    'ee_features': 'ElectronegativityDiff',
    'end_features': 'EwaldEnergy',
    'jcfid_features': 'JarvisCFID',
    'md_features': 'Meredig',
    'mpe_features': 'MaximumPackingEfficiency',
    'ofm_features': 'OrbitalFieldMatrix',
    'os_features': 'OxidationStates',
    'sc_features': 'StructuralComplexity',
    'scm_features': 'SineCoulombMatrix',
    'sh_features': 'StructuralHeterogeneity',
    'vo_features': 'ValenceOrbital',
    'xrd_features_pattern_length-20': 'XRDPowderPattern',
    'yss_features': 'YangSolidSolution'
}