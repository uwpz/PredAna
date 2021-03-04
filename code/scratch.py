

from sklearn.metrics import *


def tmp_auc(y_true, y_pred):
    #pdb.set_trace()
    print(len(y_true))
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
    return roc_auc_score(y_true, y_pred, multi_class="ovr")
