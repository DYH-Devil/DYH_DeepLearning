from Encoder import Encoder
from Decoder import Decoder
from dataset import dataloader

encoder = Encoder()
decoder = Decoder()
for input , label , input_len , label_len in dataloader :
    out , hidden , _ = encoder(input , input_len)#out_len不需要
    decoder(label , hidden)
