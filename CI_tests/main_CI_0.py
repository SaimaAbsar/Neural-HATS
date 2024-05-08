# Feb 2024
# same as main_CI_1.py but with Z = Random
# To perform marginal KCI-test for real data

import argparse
import time
import numpy as np
import pandas as pd
import trainer_CI_0 as trainer
from PyRKHSstats.kcit import perform_monte_carlo_kcit
from PyRKHSstats.kernel_wrapper import KernelWrapper
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF 
from utils import *

# Perform the KCI test
def kcit_test(data_x, data_y, data_z):
    # Kernels to use
    length_scale_kx = np.median(np.abs(pdist(data_x)))
    kernel_kx = KernelWrapper(RBF(length_scale=length_scale_kx))
    length_scale_ky = np.median(np.abs(pdist(data_y)))
    kernel_ky = KernelWrapper(RBF(length_scale=length_scale_ky))
    length_scale_kz = np.median(np.abs(pdist(data_z)))
    kernel_kz = KernelWrapper(RBF(length_scale=length_scale_kz))

    test = perform_monte_carlo_kcit(data_x, data_y, data_z, kernel_kx, kernel_ky, kernel_kz,
                epsilon=1e-1, test_level=0.1)
    return test

# Create combinations of (x,y)
def comb(num_nodes):
    prod = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i==j: continue
            prod.append((i,j))
    return prod

def convert_adj_to_CI(adj):
    CI = np.zeros([adj.shape[0], adj.shape[1]])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i,j] == 0:
                CI[i,j] = 1
    return CI

# Form the CI-matrix
def prepare_CI_table(data, epochs, num_nodes, G):
    CI_table = np.zeros([num_nodes, num_nodes]) # Intialize the CI matrix assuming all dependent

    # test for every pair, given every node z
    prod = comb(num_nodes)
    for (x,y) in prod:    
        h_x,h_z,h_y = trainer.train(data, epochs=epochs, x_values=x, y_values=y) # X -> Y|Z 
        print("\nTesting for (%d --> %d| Random)" %(x,y))
        statistics = kcit_test(h_x, h_y, h_z)  
        tci, thresh, reject = statistics['TCI'], statistics['Rejection threshold'], statistics['Reject H0 (H0 : X _||_ Y | Z)']
        print("tci, threshold: ", tci, thresh)
        print('Reject H0 (H0 : X _||_ Y | Z): ', reject)
        open_path = bool(G[x,y])
        print("Mu-PC open route: ", open_path)
        if reject==False:  #open_path: # x and y independent given z
            CI_table[x][y] = True   # put 1 if independent a/c to definition of M
            #break
   
    return CI_table

# Perform independent test for every pair (x,y) given z
def ind_test(data, epochs, num_nodes, G):
    iter = 0
    correct = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # test for every pair, given every node z
    prod = comb(num_nodes)
    for (x,y) in prod:    
        h_x,h_z,h_y = trainer.train(data, epochs=epochs, x_values=x, y_values=y) # X -> Y|Z 
        print("\nTesting for (%d --> %d| Random)" %(x,y))
        statistics = kcit_test(h_x, h_y, h_z)  
        tci, thresh, reject = statistics['TCI'], statistics['Rejection threshold'], statistics['Reject H0 (H0 : X _||_ Y | Z)']
        print("tci, threshold: ", tci, thresh)
        print('Reject H0 (H0 : X _||_ Y | Z): ', reject)
        open_path = bool(G[x,y])
        print("Mu-PC open route: ", open_path)
        if reject==open_path: # x and y independent given z  #reject==False:
            correct += 1
            if reject== True and open_path == True: TN += 1     # reject True means dependant => 0 in CI-mat
            else: TP += 1
        else:
            if reject== True and open_path == False: FN += 1 
            elif reject== False and open_path == True: FP += 1
        iter += 1
    acc = (correct/iter)*100
    prec = (TP/(TP+FP))*100
    rec = (TP/(TP+FN))*100
    return acc, prec, rec 


st = time.time() #start time
if __name__ == '__main__':    
    # Need to provide the input data directory as an argument
    parser = argparse.ArgumentParser("input info")
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--graph", type=str, help="path to ground-truth graph")
    parser.add_argument("--output", type=str, help="path to output directory")
    args = parser.parse_args()
    data_dir = args.data
    gr_dir = args.graph
    out_dir = args.output
    # Input-data
    x = pd.read_csv(data_dir,header=None,sep=',').to_numpy()
    data = np.array(x)
    data = data[:500]   # Use only first 500 samples from the dataset
    print("data shape: ", data.shape)
    d = np.shape(data)[1]  # number of variables
    epochs = 600
    # form the input graph
    matrix = np.load(gr_dir)  # path to the input graph
    print("A true:\n", matrix)
    #g = pd.DataFrame(matrix) 
    
    # To obtain the CI matrix
    CI_gt = convert_adj_to_CI(matrix) #find_ground_truth_CI(num_nodes=d, G=g)
    #print('Ground truth CI Matrix:\n', CI_gt)
    CI_table = prepare_CI_table(data, epochs, num_nodes=d, G=matrix)
    np.savetxt(out_dir, CI_table, delimiter=',')    # path to save the CI matrix
    print('Ground truth CI Matrix:\n', CI_gt)
    print('CI_table:\n', CI_table)
    compute_results(A_true=CI_gt, A_pred=CI_table)

    # Perform all CI tests
    '''acc, precision, recall = ind_test(data, epochs, num_nodes=d, G=matrix)
    print("accuracy = ", acc)
    print('Precision: ', precision)
    print('Recall: ', recall)
    f1 = 2*((precision*recall)/(precision+recall))
    print('F1-score: ', f1)
'''
# Get the end time
et = time.time()
# Get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')