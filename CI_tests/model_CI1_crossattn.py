# Aug 2023
# model based on LSTM and cross-attention for x,y,z

import numpy as np
import torch
torch.manual_seed(0)
from torch import nn
import torch.nn.functional as F
np.random.seed(0)
torch.set_default_dtype(torch.float32)
torch.cuda.empty_cache()

is_cuda = torch.cuda.is_available()
if is_cuda: device = torch.device("cuda:0")
else: device = torch.device("cpu")


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, batch_size, \
                 window_size, data_len):
        super(Model, self).__init__()

        # Defining some parameters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.atten_dim = 2*hidden_dim
        self.output_size = output_size

        #Defining the layers
        # LSTM Layer
        self.lstm_in = nn.LSTM(input_size, self.hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.2)  
        self.lstm_out = nn.LSTM(input_size, self.hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.2)  

        # Attention layers
        self.W_key = nn.Linear(self.hidden_dim, self.atten_dim)
        self.W_value = nn.Linear(self.hidden_dim, self.atten_dim)
        self.W_query = nn.Linear(self.hidden_dim, self.atten_dim)

        self.softmax = nn.Softmax(dim=1)

        # MLP layers
        self.fc = nn.Sequential(
          nn.Linear(self.atten_dim, self.output_size), #bias=False),
          nn.Sigmoid())

    def compute_attention(self, hxz, hy):
        key = self.W_key(hxz) # [batch,2,atten_dim]
        value = self.W_value(hxz)  # [batch,2,atten_dim]
        query = self.W_query(hy)
        #print("key:", key.shape)
        #print("query:", query.shape)
        #print("value:", value.shape)
        score = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.atten_dim)
        #print("score:", score.shape)
        attn = F.softmax(score, -1)     #[batch,1,2]
        #print("att: ", attn.shape)
        context = torch.bmm(attn, value)    #[batch,1,atten_dim]
        #print("context:", context.shape)
        return context, attn    #[batch, 2, atten_dim], #[batch, 2, 2]


    def forward(self, input_x, input_z, input_y):    

        # LSTM
        lstm_outx, _ = self.lstm_in(torch.unsqueeze(input_x,2)) # last layer of hidden state at all time points [batch,L,h_out]
        lstm_outz, _ = self.lstm_in(torch.unsqueeze(input_z,2))
        lstm_outy, _ = self.lstm_out(torch.unsqueeze(input_y,2))

        h_x = lstm_outx[:,-1:,:]  # last time point hidden state [batch,h_out]
        h_z = lstm_outz[:,-1:,:]
        h_y = lstm_outy[:,-1:,:]

        # Attention computation
        h_xz = torch.cat((h_x, h_z), dim=1)
        context, _ = self.compute_attention(h_xz, h_y)  

        d = torch.squeeze(context)   # [batch,atten_dim]  
        out_MLP = self.fc(d)
        #print(out_MLP.shape)

        return out_MLP, torch.squeeze(h_x.clone()), torch.squeeze(h_y.clone()), torch.squeeze(h_z.clone())


    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers,batch_size,self.hidden_dim_dec).to(device)
        return (torch.nn.parameter.Parameter(hidden, requires_grad=True), torch.nn.parameter.Parameter(hidden, requires_grad=True))  