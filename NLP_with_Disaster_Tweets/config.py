import pickle
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ws = pickle.load(open('./model/ws.pkl' , 'rb'))
hidden_size = 256
embedding_dim =300
num_layers = 4
drop_out = 0.2
max_len = 150