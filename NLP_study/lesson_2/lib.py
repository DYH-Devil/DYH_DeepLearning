import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#定义一些数值
K = 100#负采样样本数，出现1个正值必须出现100个负值
C = 3#中心词窗口大小
NUM_EPOCH = 2#训练次数
MAX_VOCAB_SIZE = 30000#词典中的词数
BATCH_SIZE = 128
LEARNING_RATE = 0.2#学习率
EMBEDDING_SIZE = 100#embedding维数

