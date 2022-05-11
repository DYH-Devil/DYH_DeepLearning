import torch
from model import LSTM
from dataset import features , labels
import lib
import os
from torch.optim import Adam
from plotly import graph_objects as go

lstm_model = LSTM().to(lib.device)
optimizer = Adam(lstm_model.parameters() , lr = 0.01)

if os.path.exists('./model/model.pkl') :
    lstm_model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))


y_true = labels.cpu().numpy().squeeze()#从GPU上取下来

y_predict = lstm_model(features)
y_predict = y_predict.detach().cpu().numpy().squeeze()#从GPU上取下来

fig = go.Figure()
fig.add_trace(go.Scatter(y = y_predict , name = 'y_predict'))
fig.add_trace(go.Scatter(y = y_true , name = 'y_true'))

fig.show()
