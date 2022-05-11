import math

import torch
from sklearn.metrics import mean_squared_error
from train import x_train , x_test , y_train , y_test  , model , train
from sklearn.preprocessing import MinMaxScaler
from feature_make import scaler
import pandas as pd
import numpy as np



y_test_pred = model(x_test)#测试集预测值

y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
testScore = math.sqrt(mean_squared_error(y_test , y_test_pred))
print("Test scoree" , testScore)