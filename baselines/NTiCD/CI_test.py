
import torch
from utils import *

is_cuda = torch.cuda.is_available()
if is_cuda: device = torch.device("cuda:0")
else: device = torch.device("cpu")

def CI_test_Loss(weight, CI_table):
    #weight = torch.from_numpy(weight)
    #num_nodes = weight.shape[0]
    CI_table = torch.from_numpy(CI_table).to(device)
    mat = CI_table.mul(weight)
    return torch.sum(mat * mat) #torch.norm(mat) 