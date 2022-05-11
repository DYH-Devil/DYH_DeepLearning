import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
import config

class seq2seq_Model(nn.Module) :
    def __init__(self):
        super(seq2seq_Model, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self , input , input_len , label , label_len):
        encoder_output , encoder_hidden = self.encoder(input , input_len)
        decoder_output , decoder_hidden = self.decoder(label , encoder_hidden)
        return decoder_output , decoder_hidden
