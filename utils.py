import os
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

def load_csv(path):
    save_path = os.path.join(os.getcwd(), path)
    open_file = open(save_path, 'rb')
    labels_df = pickle.load(open_file)
    open_file.close()
    return labels_df

def bootstrap_roc_auc(num_samples, y, preds):
    n = len(y)
    auc_scores = np.zeros(num_samples)
    for i in range(num_samples):
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        y_sample = y[indices]
        preds_sample = preds[indices]
        auc_scores[i] = roc_auc_score(y_sample, preds_sample)
    
    estimated_mean = np.mean(auc_scores)
    estimated_std = np.std(auc_scores, ddof=1)

    return estimated_mean, estimated_std
