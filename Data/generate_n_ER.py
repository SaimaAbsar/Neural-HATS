# Oct 2023

import numpy as np
import igraph as ig
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

def generate_ER_graph(n,m):
    G_und = ig.Graph.Erdos_Renyi(n=n, m=m, directed=True, loops=False)
    G = np.array(G_und.get_adjacency().data)
    print("adjacency:\n", G)
    return G


def generate_timeseries(n, d, noise_scale, adjacency_matrix):
    # Initialize the time series data
    time_series_data = np.zeros((n, d))
    
    # Generate random positive regression coefficients from a uniform distribution
    coefficients = np.random.uniform(low=0.1, high=2.0, size=(d, d, 5))

    # Generate initial values for the time series
    time_series_data[:5, :] = np.random.normal(0, 1, (5, d))
    
    # Generate the time series data using the adjacency matrix and coefficients
    for t in range(5, n):
        for i in range(d):
            influence = 0
            for j in range(d):
                # Apply the influence of variable j on variable i
                if adjacency_matrix[j, i] == 1:
                    for lag in range(1, 6):
                        influence += coefficients[i, j, lag-1] * np.cos(time_series_data[t-lag, j] + 1)
            # Add noise
            noise = noise_scale * np.random.randn()    #np.random.normal()
            # Update the time series value
            time_series_data[t, i] = influence + noise

    return time_series_data


# Example usage
if __name__ == '__main__':
    n = 1000  # Number of time steps
    d = 6     # Number of variables
    m = 7     # Probability for edge creation in ER graph
    noise_scale = 0.01

    # Generate the ER graph adjacency matrix
    A = generate_ER_graph(d, m=7)

    # Generate the time series data
    T = generate_timeseries(n, d, noise_scale, A)

    # Save the adjacency matrix and the generated time series data
    np.save('A_true_ER_test', A)
    np.savetxt('6_ER_test.txt', T, delimiter=',')

    

# Note: this file generates non-linear (cosine) data with its ground truth adjacency matrix
# the adjacency matrix is saved as a numpy array of shape: (d x d), where d = #Variables
# the adjacency matrix is generated from the ER graph model
# generated data is of shape: (n x d), where n = #time-steps, d = #Variables; 
# with no index and header, comma-seperated file


