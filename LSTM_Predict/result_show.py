"""
将结果反映在图上
"""

import torch
from plotly import graph_objects as go
from LSTM_Model_3 import LSTM_Model
from dataset_2 import features , labels
import lib
import os
from torch.optim import Adam
from plotly import graph_objects as go

# 预测验证预览
labels = labels.cpu().numpy().squeeze()
lstm_model = LSTM_Model()
optimizer = Adam(lstm_model.parameters() , lr = 0.01)
lstm_model = lstm_model.to(lib.device)

if os.path.exists('./model/model.pkl') :
    lstm_model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

y_pred = lstm_model(features).detach().cpu().numpy().squeeze()
fig = go.Figure()
fig.add_trace(go.Scatter(y=labels, name='y_true'))
fig.add_trace(go.Scatter(y=y_pred, name='y_pred'))
fig.show()
