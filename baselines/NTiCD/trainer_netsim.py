
# trainer code for simultaneous training of LSTM-GCN-MLP
# updated to add CI-constraint to convert to hybrid method

import numpy as np
import torch
import os
from torch import nn
torch.cuda.empty_cache()
from model_with_GCN import Model
from utils import *
from CI_test import *  
torch.manual_seed(1230)
np.random.seed(1230)
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
#import wandb
torch.autograd.set_detect_anomaly(True)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
if is_cuda: device = torch.device("cuda:0")
else: device = torch.device("cpu")

# Preprocess data to form a matrix
def preprocess(data, n, window_size):
    #Normalize data column-wise
    sc = MinMaxScaler(feature_range=(0, 1))
    training_data = sc.fit_transform(data)
    # structuring the data 
    target = [] 
    in_seq = []
    for i in range(n-window_size-1):
      list1 = []
      for j in range(i,i+window_size):
           list1.append(training_data[j])
      in_seq.append(list1)
      target.append(training_data[j+1])
    return np.array(in_seq), np.array(target)

class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32).to(device)
        self.y = torch.tensor(y,dtype=torch.float32).to(device)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len

def init_weights(m):
    if isinstance(m, nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight)


### Training the 3 models ###
def train(path, is_saved, regularization_param, epochs, CI_path):
    print('\nTraining started...\n')

    # Input-data
    x = pd.read_csv(path,sep=',').to_numpy()
    data = np.array(x)
    data = data
    #print(data.shape)
    d = np.shape(data)[1]  # number of variables
    n = np.shape(data)[0]  # number of time-steps

    # =====================================================================
    # CI constraint added here
    # =====================================================================
    #g = pd.DataFrame(matrix) 
    CI_table = np.loadtxt(CI_path, delimiter=',')
    print('CI table:\n', CI_table)
    # =====================================================================
    
    # Define NTiCD hyperparameters
    window_size = 5
    input_size = 1 
    hidden_dim = 50
    n_layers = 2
    lr=1e-3
    epochs = epochs
    l1_lambda = regularization_param
    batch_size = 128
    lambda_CI = 5e-3
    x_train, y_train = preprocess(data, n, window_size)
    dataset = timeseries(x_train,y_train)
    train_loader = DataLoader(dataset,shuffle=True,batch_size=batch_size, drop_last=True)

    # define the model
    model = Model(input_size=input_size, output_size=input_size, hidden_dim=hidden_dim, \
        n_layers=n_layers, n=d, batch_size=batch_size).to(device)

    filename1 = 'modelLSTM_GCN_MLP' 
    PATH1 = os.path.join(r'./saved_models/', filename1)
    if is_saved == 1: model.load_state_dict(torch.load(PATH1))  # can be used to start training from the saved model
    else: model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.9)

    criterion = torch.nn.MSELoss()
    L = []

    # train to update A
    for epoch in range(epochs):
        total_loss = 0
        e = 0
        model.train()
        for _, (in_seq, target) in enumerate(train_loader):
            optimizer.zero_grad()
            model.hidden = (torch.zeros(n_layers,batch_size,hidden_dim).to(device),
                            torch.zeros(n_layers,batch_size,hidden_dim).to(device))
            out = model(in_seq) # returns the output from final MLP layer

            # Calculate error
            mse = criterion(out, target)
            l1_norm = torch.norm(model.A_prime)     # regularization to add sparsity
            loss2 = l1_lambda * l1_norm     # regularization to add sparsity
            l3 =  l1_lambda * torch.sum(torch.square(model.A_prime-torch.mean(model.A_prime,0)))    # regularization
            CI_loss = CI_test_Loss(model.A_prime, CI_table)  #   CI_table = M: conditional-independance testing to add constraint
            
            loss = mse + loss2 - l3 + lambda_CI*CI_loss        # Basically, loss = mse + l1_lambda*l1_norm - l1_lambda*l3 + lambda_CI*CI_loss
            loss.backward()    # Does backpropagation and calculates gradients
            
            # Updates the weights accordingly
            optimizer.step() 

            # to keep a record of total loss per epoch
            total_loss += loss.item()
            e += 1

        avg_loss = total_loss/e
        
        if epoch%100 == 0:
            print('MSE: ', mse)
            print('Epoch: {} .............'.format(epoch), end=' ')
            print("Loss: {:.4f}".format(avg_loss))
        L.append(avg_loss)
        epoch_scheduler.step()

    torch.save(model.state_dict(), os.path.join(r'./saved_models/', 'modelLSTM_GCN_MLP'))   # save the trained model
    A_pred = model.A_prime

    print('\nFinished training.\n')
    print(A_pred)
    torch.save(model.A.data,'A_pred.pt')
    return A_pred.data


    