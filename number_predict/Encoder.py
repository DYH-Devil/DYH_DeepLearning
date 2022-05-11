import torch.nn as nn
import config
from torch.nn.utils.rnn import pack_padded_sequence , pad_packed_sequence

class Encoder(nn.Module) :
    def __init__(self) :
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings =len(config.num_sequence) , embedding_dim = config.embedding_dim , padding_idx = config.num_sequence.PAD)
        #padding_idx:int,填充字符在索引中的位置,使用后该字符不计算梯度
        self.gru = nn.GRU(input_size = config.embedding_dim , num_layers = config.num_layers , hidden_size = config.hidden_size , batch_first = True , bidirectional = False)

    def forward(self , input , input_len) :
        """
        :param input: [batchsize , seqlen]
        :return:
        """
        embeded = self.embedding(input)#embedding
        #embeded:[batchsize , seqlen , embedding_dim]
        embeded = pack_padded_sequence(embeded , input_len , batch_first = True)#对embedding后的数据进行填充打包,意义在于每条batch中的数据填充，使每条数据长度一致
        #参数:input,input_len,batch_first，total_len
        #注意:在pack_padded_sequence之前，batch中的数据必须先按长度进行降序排列。这也是为什么在collate_fn里按长度排列input的原因
        #total_len:将input的数据填充到指定长度，不得小于seqlen,如设为None,则取batch中的最长(按长度降序后的第一个)作为total_len
        out , hidden = self.gru(embeded)#GRU
        #out:[batchsize , seqlen , hiddensize]
        #hidden:[layers * bidirectioanl , batch , hiddensize]
        out , out_len= pad_packed_sequence(out , batch_first = True , padding_value = config.num_sequence.PAD)#对out解包,注意，解包后返回的是两个值,output和其长度
        return out , hidden


if __name__ == '__main__':
    encoder = Encoder()#实例化
    from dataset import dataloader
    for input , label , input_len , label_len in dataloader :
        out , hidden  = encoder(input , input_len)
        print(out.size())
        print(hidden.size())
        break