
# Similar to trainer_CI_1.py except z = random variable

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
import numpy as np
import torch
from torch import nn
torch.cuda.empty_cache()
from model_CI1_crossattn import Model
torch.manual_seed(0)
np.random.seed(0)
import random
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()


is_cuda = torch.cuda.is_available()
if is_cuda: device = torch.device("cuda:0")
else: device = torch.device("cpu")

# Preprocess data to form a matrix
def preprocess(data, n, window_size, out_window, x_values, y_values):
    #Normalize data column-wise
    sc = MinMaxScaler(feature_range=(0, 1))
    training_data = sc.fit_transform(data)
    x = training_data[:, x_values]
    y = training_data[:, y_values]
    # structuring the data 
    target = [] 
    in_seq = []
    for i in range(n-window_size-out_window-1):
      list1 = []
      list2 = []
      for j in range(i,i+window_size):
           list1.append(x[j])
      in_seq.append(list1)
      for k in range(j, j+out_window):
          list2.append(y[k])
      target.append(list2)
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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

### Training the models ###
def train(data, epochs, x_values, y_values): 
    print('\nTraining started...\n')
    print("x, y = ", x_values, y_values)
    data_len = np.shape(data)[0]  # number of time-steps
    
    window_size = 5
    out_window = 1
    input_size = 1
    hidden_dim = 50
    n_layers = 2
    lr = 1e-3
    epochs = epochs
    batch_size = 128
    x_train, y_train = preprocess(data, data_len, window_size, out_window, x_values, y_values)
    g = torch.Generator()
    g.manual_seed(0)
    dataset = timeseries(x_train,y_train)
    train_loader = DataLoader(dataset,shuffle=True,batch_size=batch_size, drop_last=True, worker_init_fn=seed_worker, generator=g)

    # define the model
    model = Model(input_size=input_size, output_size=out_window, hidden_dim=hidden_dim, \
        n_layers=n_layers, batch_size=batch_size, window_size=window_size, data_len = data_len).to(device)
 
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.9)

    criterion = torch.nn.MSELoss()
    L = []

    # train to update A
    for epoch in range(epochs):
        total_loss = 0
        iteration = 0
        model.train()
        atten_len = (data_len-window_size-1)//batch_size
        H_x = np.zeros(shape=(atten_len*batch_size, hidden_dim))
        H_z = np.zeros(shape=(atten_len*batch_size, hidden_dim))
        H_y = np.zeros(shape=(atten_len*batch_size, hidden_dim))
        for idx, (in_seq, target) in enumerate(train_loader):
            optimizer.zero_grad()
            input_x = in_seq
            z = np.random.normal(size=(batch_size, window_size))
            input_z = torch.tensor(z,dtype=torch.float32).to(device)    # to test if CI(X --> Y|random_var)
            out, h_x,h_y,h_z = model(input_x, input_z, target) # returns the output from final MLP layer
            H_x[idx*batch_size: idx*batch_size+batch_size, :] = h_x.data.cpu().numpy()
            H_z[idx*batch_size: idx*batch_size+batch_size, :] = h_z.data.cpu().numpy()
            H_y[idx*batch_size: idx*batch_size+batch_size, :] = h_y.data.cpu().numpy()

            # Calculate error
            mse = criterion(out, target)
            loss = mse 
            loss.backward()    # Does backpropagation and calculates gradients
            
            # Updates the weights accordingly
            optimizer.step() 

            # to keep a record of total loss per epoch
            total_loss += loss.item()
            iteration += 1
            
        avg_loss = total_loss/iteration
        
        if epoch%100 == 0:
            print('MSE: ', mse)
            #print('l2_norm: ', l2_norm)
            print('Epoch: {} .............'.format(epoch), end=' ')
            print("Loss: {:.4f}".format(avg_loss))
        L.append(avg_loss)

        epoch_scheduler.step()
    print(H_x.shape)

    print('\nFinished training.')

    return H_x, H_z, H_y    
