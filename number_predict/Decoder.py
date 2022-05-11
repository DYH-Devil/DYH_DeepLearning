import torch
import torch.nn as nn
import config
import torch.nn.functional as F
from dataset import dataloader

class Decoder(nn.Module) :

    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = len(config.num_sequence) ,
                                      embedding_dim = config.embedding_dim ,
                                      padding_idx = config.num_sequence.PAD)

        self.gru = nn.GRU(input_size = config.embedding_dim ,
                          batch_first = True ,
                          num_layers = config.num_layers ,
                          hidden_size = config.hidden_size)

        self.fc = nn.Linear(config.hidden_size , len(config.num_sequence))#将输出结果映射为n个数字结果，转化为vocab_size个分类问题(用于第4步)

    def forward(self , target , encoder_hidden) :
        # 1.获取encoder的输入作为第一个时间步的hidden_state
        decoder_hidden = encoder_hidden
        # 2.准备第一个时间步的input输入[batchsize , 1]SOS开始符
        batch_size = target.size(0)#最后一次可能不是config中定义的batch_size
        decoder_input = torch.LongTensor(torch.ones([batch_size , 1] , dtype = torch.int64) * config.num_sequence.SOS)
        # 3.在第一个时间步上进行计算，得出第一个时间步上的输出output和hidden_state


        # 4.把第一个时间步上的输出进行计算，得到第一个时间步上最终结果
        # 5.把前一个时间步上的hidden_state作为下一个时间步上的hidden_state,把前一个时间步上的输出output作为下一个时间步上的输入
        # 6.循环4-5
        decoder_outputs = torch.zeros([batch_size , config.max_len + 2 , len(config.num_sequence)])#用于保存每个时间步上的输出output[batch,max_len+2,vocab_size]
        for t in range(config.max_len + 2) :#+2因为有EOS和SOS
            decoder_output_t , decoder_hidden = self.forward_step(decoder_input , decoder_hidden)
            #decoder_output_t:当前时间步输出:[batch , vocab_Size]
            #decoder_hidden:当前时间步hidden
            decoder_outputs[: , t , :] = decoder_output_t#记录t时间步上的输出
            value , index = torch.topk(decoder_output_t , 1)#取第一个(softmax值最大的那个作为output结果),index是其位置
            decoder_input = index#将上一时间步的输出作为下一时间步的输入
        return decoder_outputs , decoder_hidden


    def forward_step(self , decoder_input , decoder_hidden):#计算一个时间步上的output和hidden
        """
        target:通过decoder_input,decoder_hidden得到结果res , res:[batch_size , vocab_size]转化为分类结果
        #vocab_size:0-9数字类别
        :param decoder_input: [batch_size , 1]
        :param decoder_hidden: [1 , batch_size , hidden_size]
        """
        #decoder_input是SOS字符，因此需要先embedding操作
        embedding = self.embedding(decoder_input)#embedding[batch_size , 1 , embedding_dim]
        out , hidden = self.gru(embedding , decoder_hidden)
        #out:[batch_size , 1 , hidden_size]
        #hidden:[1 , batch_size , hidden_size]
        out = out.squeeze(1)#out:[batch_size , hidden_size]
        out = self.fc(out)#out:[batch , vocab_size]
        output = F.log_softmax(out , dim = 1)#计算概率
        #print(output.size())#[batch,vocab_size]
        return output , hidden

if __name__ == '__main__':
    from Encoder import Encoder
    encoder = Encoder()
    decoder = Decoder()
    for idx, (input, label, input_len, label_len) in enumerate(dataloader):
        encoder_out , encoder_hidden = encoder(input , input_len)
        decoder_out , decoder_hidden = decoder(label , encoder_hidden)
        print(decoder_hidden.size())
        break
