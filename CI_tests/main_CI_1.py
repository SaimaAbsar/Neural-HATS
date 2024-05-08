# Neural-HATS 
# This code is used to determine the CI matrix using KCI test

import argparse
import time
import numpy as np
import pandas as pd
import trainer_CI_1 as trainer
from PyRKHSstats.kcit import perform_monte_carlo_kcit
from PyRKHSstats.kernel_wrapper import KernelWrapper
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF
from find_mu_path import open_route
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

# Determine the gt CI matrix
def find_ground_truth_CI(num_nodes, G):
    CI_gt = np.zeros([num_nodes, num_nodes])
    prod = comb(num_nodes)
    for (x,y) in prod:   
        for z in range(num_nodes):
            if z==x or z==y:
                continue
            open_path = open_route(n=num_nodes, graph=G, p=(x,y), C_=[z])
            if open_path==False:    # x and y are dependent given z
                CI_gt[x][y] = True
                break
    print('CI_gt:\n',  CI_gt)
    return CI_gt

# Form the CI-matrix using KCI-test for every pair (x,y) given z
def prepare_CI_table(data, epochs, num_nodes, G):
    CI_table = np.zeros([num_nodes, num_nodes])
    # test for every pair, given every node z
    prod = comb(num_nodes)
    for (x,y) in prod:    
        for z in range(num_nodes):
            if z==x or z==y:
                continue
            h_x,h_z,h_y = trainer.train(data, epochs=epochs, x_values=(x,z), y_values=y) # X -> Y|Z 
            print("\nTesting for (%d --> %d| %d)" %(x,y,z))
            test = kcit_test(h_x, h_y, h_z)  
            print("tci, threshold: ", test['TCI'], test['Rejection threshold'])
            print('Reject H0 (H0 : X _||_ Y | Z): ', test['Reject H0 (H0 : X _||_ Y | Z)'])
            open_path = open_route(n=num_nodes, graph=G, p=(x,y), C_=[z])
            print("Mu-PC open route: ", open_path)
            if test['Reject H0 (H0 : X _||_ Y | Z)']==False:  #open_path: x and y independent given z
                CI_table[x][y] = True   # put 1 if independent a/c to definition of M
                break
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
        for z in range(num_nodes):
            if z==x or z==y:
                continue
            #print("\nTesting for (%d --> %d| %d)" %(x,y,z))
            h_x,h_z,h_y = trainer.train(data, epochs=epochs, x_values=(x,z), y_values=y) # X -> Y|Z 
            print("\nTesting for (%d --> %d| %d)" %(x,y,z))
            statistics = kcit_test(h_x, h_y, h_z)  
            tci, thresh, reject = statistics['TCI'], statistics['Rejection threshold'], statistics['Reject H0 (H0 : X _||_ Y | Z)']
            print("tci, threshold: ", tci, thresh)
            print('Reject H0 (H0 : X _||_ Y | Z): ', reject)
            open_path = open_route(n=num_nodes, graph=G, p=(x,y), C_=[z])
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
    data = data[:500]   # We use first 500 samples from the data for performance efficiency
    print("data shape: ", data.shape)
    d = np.shape(data)[1]  # number of variables
    epochs = 600
    # form the input graph
    matrix = np.load(gr_dir)  # path to the input graph
    print("A true:\n", matrix)
    g = pd.DataFrame(matrix) 
    CI_gt = find_ground_truth_CI(num_nodes=d, G=g)
    #print('Ground truth CI Matrix:\n', CI_gt)
    
    # To obtain the CI matrix
    CI_table = prepare_CI_table(data, epochs, num_nodes=d, G=matrix)
    np.savetxt(out_dir, CI_table, delimiter=',')    # provide the path to save the CI matrix
    print('Ground truth CI Matrix:\n', CI_gt)
    print('CI_table:\n', CI_table)
    compute_results(A_true=CI_gt, A_pred=CI_table)

    # Perform all CI tests
    '''acc, precision, recall = ind_test(data, epochs, num_nodes=d, G=g)
    print("accuracy = ", acc)
    print('Precision: ', precision)
    print('Recall: ', recall)
    f1 = 2*((precision*recall)/(precision+recall))
    print('F1-score: ', f1)'''

# Get the end time
et = time.time()
# Get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')