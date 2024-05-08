# Nov 2023
# uti

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score

def compute_results(A_true, A_pred):
    acc = np.sum(A_pred == A_true)/(len(A_true)*len(A_true)) * 100 
    #shd = np.count_nonzero(A_true!=A_pred)
    precision = precision_score(A_true.flatten(),A_pred.flatten()) * 100 
    recall = recall_score(A_true.flatten(),A_pred.flatten()) * 100 
    f1 = f1_score(A_true.flatten(),A_pred.flatten()) * 100 
    print('\nPercentage accuracy: ', acc)
    #print('Current shd: ', shd)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-score: ', f1)

def remove_selfloops(A):
    ind = np.diag_indices(A.shape[0])
    A[ind[0], ind[1]] = torch.zeros(A.shape[0])
    return A

def remove_selfloops2(A):
    ind = np.diag_indices(A.shape[0])
    A[ind[0], ind[1]] = torch.ones(A.shape[0])
    return A
    
def eval_causal_structure_binary(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if not diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        precision = precision_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        recall = recall_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        accuracy = accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        bal_accuracy = balanced_accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        f1 = f1_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
    else:
        precision = precision_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        recall = recall_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        accuracy = accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        bal_accuracy = balanced_accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        #f1 = f1_score(y_true=a_true, y_pred=a_true)
        f1 = (2*precision*recall)/(precision+recall)

    print("\nAcc.: " + str(np.round(accuracy, 4)) + "; Bal. Acc.: " +
                      str(np.round(bal_accuracy, 4)) + "; Prec.: " + str(np.round(precision, 4)) + "; Rec.: " +
                      str(np.round(recall, 4)) + "; F1.: " + str(np.round(f1, 4)), end='\n')    
    #return accuracy, bal_accuracy, precision, recall, f1