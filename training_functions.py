
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
import numpy as np
from utils import bootstrap_roc_auc
from data_preprocessing import *
from sklearn.metrics import roc_auc_score


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


    plot_ax(axs[0, 1], np.log10(df['diffusion_mean_cm2_s']).to_numpy(), aggregated_preds, indexes)
    axs[0, 1].set_ylabel('probability')
    axs[0, 1].set_xlabel('diffusion coefficient, $\log_{10}D[\\frac{cm^2}{s}]$')
    axs[0, 1].set_title('distribution of np.log10(diffusivity) for aggregated preds')


    plot_ax(axs[1, 0], np.log10(df['sigma_S_cm']).to_numpy(), single_preds, indexes)
    axs[1, 0].set_ylabel('probability')
    axs[1, 0].set_xlabel('electrolytic conductivity, $\log_{10}\sigma[\\frac{\S}{cm}]$')
    axs[1, 0].set_title('distribution of np.log10(conductivity) for single preds')


    plot_ax(axs[1, 1], np.log10(df['sigma_S_cm']).to_numpy(), aggregated_preds, indexes)
    axs[1, 1].set_ylabel('probability')
    axs[1, 1].set_xlabel('electrolytic conductivity, $\log_{10}\sigma[\\frac{\S}{cm}]$')
    axs[1, 1].set_title('distribution of np.log10(conductivity) for aggregated preds')

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
    
    if X_mpdb is not None:
        preds_mpdb = np.zeros(X_mpdb.shape[0])
        
    train_roc_auc = np.zeros(len(y))

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        positive_weights = weights_for_positive[train_index]
        negative_weights = weights_for_negative[train_index]

        X_train = X[train_index, :]
        y_train = y[train_index]

        X_scaled_positive = X_train[positive_weights > threshold]
        X_scaled_negative = X_train[negative_weights > threshold]
 
        labels = np.hstack((np.ones(X_scaled_positive.shape[0]), np.zeros(X_scaled_negative.shape[0])))
        weights = np.hstack([positive_weights[positive_weights > threshold], negative_weights[negative_weights > threshold]])
        X_scaled = np.vstack([X_scaled_positive, X_scaled_negative])
        model = model_class

        if isinstance(model, CatBoostClassifier):
            train_pool = Pool(X_scaled, label = labels, weight = weights)
            model.fit(train_pool)

        else:
            model.fit(X_scaled, labels, sample_weight = weights)

        X_test = X[test_index]
        preds[test_index] = model.predict_proba(X_test)[:, 1]
        train_roc_auc[i] = roc_auc_score(labels, model.predict_proba(X_scaled)[:, 1])

        if i == 0 and verbose == True:
            print(f'roc-auc on train for {i} fold with size {X_scaled.shape[0]}: {train_roc_auc[i]}')

        if X_mpdb is not None:
            preds_mpdb += model.predict_proba(X_mpdb)[:, 1]

    estimated_mean, estimated_std = bootstrap_roc_auc(1000, y, preds)

    if verbose == True:
        print(f'test roc-auc mean: {estimated_mean}, std: {estimated_std}')
        print(f'mean train roc-auc: {np.mean(train_roc_auc)}')
    
    if X_mpdb is not None:
        return preds, model, estimated_mean, estimated_std, np.mean(train_roc_auc), np.std(train_roc_auc), preds_mpdb / X.shape[0]
    else:
        return preds, model, estimated_mean, estimated_std, np.mean(train_roc_auc), np.std(train_roc_auc)


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


        X_train = X[train_index, :]
        y_train = y[train_index]

        model = model_class
        model_class.fit(X_train, y_train)

        X_test = X[test_index]

        preds[test_index] = model.predict_proba(X_test)[:, 1]

        preds_train = model.predict(X_train)
        train_roc_auc[i] = roc_auc_score(y_train, preds_train)
        
    estimated_mean, estimated_std = bootstrap_roc_auc(1000, y, preds)

    if verbose == True:
        print(estimated_mean, estimated_std)
        print(f'mean train roc-auc {train_roc_auc.mean()}')

    return preds, model



from misc_utils.feature_analysis import *
from misc_utils import augment_Kahle2020
from misc_utils.augment_preds import join_data_and_preds_Kahle2020


def calculate_ROClikeComparisonMetricsKahle(preds_Kahle2020,
                            predictions_kahle: np.ndarray):
    

    features_meta_info = pd.DataFrame(columns=['feature', 'level', 'type', 'weighted_direction'])
    features_meta_info = features_meta_info.set_index(features_meta_info.columns[0])

    [preds_Kahle2020], all_features = add_feature_np(
        [preds_Kahle2020],
        features_meta_info,
        values = [predictions_kahle],
        name = "training",
        type = "training",
        level=0.0,
    )

    sim_neg = len(preds_Kahle2020.query("condNE1000 <= 1e-2"))
    roclike_metrics = ROClikeComparisonMetrics()

    feature_scores_base = roclike_metrics.eval_features(
            preds_df=preds_Kahle2020,
            features_meta_info=all_features,
            num_bootstrap_samples=1000,
            num_negatives_max=0.1 * sim_neg,
            positive_on_extrap300=False,
            weight_validation_plots=False,
        )
        
    return feature_scores_base["score"]["training"]




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


featurizers_mapping = {
    'ape_features': 'AtomicPackingEfficiency',
    'bc_features': 'BandCenter',
    'bf_features': 'BondFractions',
    'co_features': 'ChemicalOrdering',
    'density_features': 'DensityFeatures',
    'ee_features': 'ElectronegativityDiff',
    'end_features': 'EwaldEnergy',
    'jcfid_features': 'JarvisCFID',
    'gsf_features': 'GaussianSymmFunc',
    'md_features': 'Meredig',
    'mpe_features': 'MaximumPackingEfficiency',
    'ofm_features': 'OrbitalFieldMatrix',
    'os_features': 'OxidationStates',
    'sc_features': 'StructuralComplexity',
    'scm_features': 'SineCoulombMatrix',
    'sh_features': 'StructuralHeterogeneity',
    'vo_features': 'ValenceOrbital',
    'xrd_features_pattern_length-20': 'XRDPowderPattern',
    'yss_features': 'YangSolidSolution',
    'SOAP_features_partialS_outer_rcut-3_nmax-5_lmax-3': 'SOAP'
}