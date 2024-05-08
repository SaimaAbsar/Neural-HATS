# NTiCD main function

import numpy as np
import torch
import argparse
torch.cuda.empty_cache()
import trainer as trainer
from utils import *
torch.manual_seed(10)
np.random.seed(10)
#from sklearn.metrics import precision_score, recall_score, f1_score
import time

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda:0")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

st = time.time() #start time
if __name__ == '__main__':    
    # Need to provide the input data directory as an argument
    parser = argparse.ArgumentParser("input info")
    parser.add_argument("--datapath", type=str, help="path to input directory")
    parser.add_argument("--graph", type=str, help="path to input graph")
    parser.add_argument("--d", type=int, help="Number of variables")
    parser.add_argument("--ci", type=str, help="path to CI matrix")
    args = parser.parse_args()
    data_dir = args.datapath
    graph = args.graph
    d = args.d
    CI_path = args.ci

    A_true = np.load(graph)
    print(A_true)
    A_pred = torch.Tensor(1*np.zeros([d,d], dtype=np.double) + np.random.uniform(low=0, high=0.1, size=(d,d)))
    torch.save(A_pred,'./A_pred.pt')  #A_pred initialized to very small values, so that sigmid(A_pred)=0.5 initially; will be updated during training

    acc = 0
    mse = []
    reg_param = 2e-3    # regularization parameter
    epochs = 1000
    # training
    A_pred = trainer.train(path=data_dir, is_saved=0, regularization_param=reg_param, epochs=epochs, CI_path=CI_path)  
    # the 'is_saved' parameter can resume training from the saved model if set to 1
    
    A_pred = A_pred.cpu().numpy()
    graph_thres = np.mean(A_pred,0) # columnwise mean to calculate different threshold for each variable
    print("Graph Threshold: ", graph_thres)

    # Evaluation
    A_pred[np.abs(A_pred) < graph_thres] = 0    
    A_pred[np.abs(A_pred) >= graph_thres] = 1
    print('A true: \n', A_true)
    print('A predicted: \n', A_pred)
    compute_results(A_true=A_true, A_pred=A_pred)
    eval_causal_structure_binary(A_true, A_pred, diagonal=False)

    print('\ncurrent lambda value: ', reg_param)

# Get the end time
et = time.time()
# Get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
print('Results of GCN sum version')