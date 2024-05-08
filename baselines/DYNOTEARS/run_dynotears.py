import pandas as pd

from dynotears import from_pandas_dynamic

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import time
np.random.seed(10)
st = time.time()

def dynotears(data,CI_table, tau_max=5, alpha=0.05):
    graph_dict = dict()
    for name in data.columns:
        graph_dict[name] = []

    #sm = from_pandas_dynamic(data, CI_table, p=tau_max, w_threshold=0.01, lambda_w=0.2, lambda_a=0.2)  # hyperparameters for ER-syn data
    sm = from_pandas_dynamic(data, CI_table, p=tau_max, w_threshold=0.05, lambda_w=0.2, lambda_a=0.2)  # for Netsim

    # print(sm.edges)
    # print(sm.pred)

    tname_to_name_dict = dict()
    count_lag = 0
    idx_name = 0
    for tname in sm.nodes:
        tname_to_name_dict[tname] = data.columns[idx_name]
        if count_lag == tau_max:
            idx_name = idx_name +1
            count_lag = -1
        count_lag = count_lag +1

    for ce in sm.edges:
        c = ce[0]
        e = ce[1]
        tc = int(c.partition("lag")[2])
        te = int(e.partition("lag")[2])
        t = tc - te
        if (tname_to_name_dict[c], -t) not in graph_dict[tname_to_name_dict[e]]:
            graph_dict[tname_to_name_dict[e]].append((tname_to_name_dict[c], -t))

    # g = sm.to_directed()
    return graph_dict

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
    return accuracy, bal_accuracy, precision, recall, f1

if __name__ == "__main__":

    parser = argparse.ArgumentParser("input info")
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--graph", type=str, help="path to ground-truth graph")
    parser.add_argument("--output", type=str, help="path to output directory")
    args = parser.parse_args()
    path = args.data
    gr_dir = args.graph
    out_dir = args.output

    # Input-data
    data = pd.read_csv(path,header=None,sep=',')
    print(data.shape)
    
    CI_table = np.loadtxt(out_dir, delimiter=',')
    print('CI-table:\n', CI_table)
    A_true = np.load(gr_dir)
    g = dynotears(data, CI_table, 1)
    A_pred = np.zeros_like(A_true)

    # use for our data with no header
    for k in g.keys():
        links = g[k]
        parent = []
        for l in links:
            parent.append(l[0])
            parent = [*set(parent)]
        #print(k, parent)
        for p in parent:
            A_pred[p,k] = 1

    print('A true: \n', A_true)
    print('A predicted: \n', A_pred)

    acc, bal_acc, precision, recall, f1 = eval_causal_structure_binary(A_true, A_pred, diagonal=False)
    print('\nPercentage accuracy: ', acc)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-score: ', f1)

# Get the end time
et = time.time()
# Get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

