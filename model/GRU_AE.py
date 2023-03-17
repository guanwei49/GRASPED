import math
from collections import Iterator

import torch.nn.functional as F
import torch
from torch import nn, multiprocessing
import random


from model import device


class Encoder(nn.Module):
    def __init__(self, input_size, enc_hidden_dim, num_layers, dec_hid_dim):
        super().__init__()
        self.input_size = input_size
        self.enc_hiddenc_dim = enc_hidden_dim
        self.num_layers = num_layers
        self.dec_hid_dim = dec_hid_dim
        self.embedding = nn.Embedding(self.input_size, self.enc_hiddenc_dim)
        self.gru = nn.GRU(input_size = enc_hidden_dim,
                          hidden_size = enc_hidden_dim,
                          num_layers = num_layers,
                          dropout=0.3, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(enc_hidden_dim * 2),
            nn.Linear(enc_hidden_dim * 2, dec_hid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, init_input):
        embedding= self.embedding(init_input)
        enc_output, hidden = self.gru(embedding,  None)
        # 将正向最后时刻的输出与反向最后时刻的输出进行拼接，得到的维度应该是[batch,enc_hiddenc_dim*2]
        # hidden[-1,:,:] 反向最后时刻的输出, hidden[-2,:,:] 正向最后时刻的输出
        h_m = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) # 在第一个维度进行拼接
        s0 = self.fc(h_m)
        return enc_output,s0  # enc_output:[seq_len,batch,enc_hiddenc_dim*2] s0:[batch,dec_hiddenc_dim]




class Attention(nn.Module):
    '''
    加性注意力
    '''
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)  # 输出的维度是任意的
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)  # 将输出维度置为1

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hidden_dim]
        # enc_output = [seq_len, batch_size, enc_hid_dim * 2]

        seq_len = enc_output.shape[0]

        # repeat decoder hidden state seq_len times
        # s = [seq_len, batch_size, dec_hid_dim]
        s = s.repeat(seq_len, 1,1)  # [batch_size, dec_hid_dim]=>[seq_len, batch_size, dec_hid_dim]

        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        attention = self.v(energy).squeeze(
            2)  # [seq_len, batch_size, dec_hid_dim]=>[seq_len，batch_size, 1] => [seq_len, batch_size]

        attention_probs=F.softmax(attention, dim=0).transpose(0, 1).unsqueeze(1)  # [batch_size, 1 , seq_len]

        enc_output = enc_output.transpose(0, 1)

        # # c = [1, batch_size, enc_hid_dim * 2]
        c = torch.bmm(attention_probs, enc_output).transpose(0, 1)

        return c,attention_probs


# class Attention(nn.Module):
#     '''
#     点积注意力
#     '''
#     def __init__(self, enc_hid_dim, dec_hid_dim):
#         super().__init__()
#         self.hidden=64
#         self.query = nn.Linear(dec_hid_dim, self.hidden)
#         self.key = nn.Linear(enc_hid_dim * 2, self.hidden)
#
#
#     def forward(self, s, enc_output):
#         # s = [batch_size, dec_hidden_dim]
#         # enc_output = [seq_len, batch_size, enc_hid_dim * 2]
#         s = s.unsqueeze(0) # [batch_size, dec_hid_dim]=>[1, batch_size, dec_hid_dim]
#         s=s.transpose(0, 1)
#         q=self.query(s)
#         enc_output=enc_output.transpose(0, 1)
#         k=self.key(enc_output)
#         k=k.transpose(1, 2)
#
#         attention_scores= torch.bmm(q, k)
#         attention_scores=attention_scores/ math.sqrt(self.hidden)
#
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)##[ batch_size, 1, seq_len]
#
#         result = torch.bmm(attention_probs, enc_output).transpose(0, 1)
#
#         return result,attention_probs


class Decoder(nn.Module):
    def __init__(self, vocab_size, enc_hid_dim, dec_hid_dim,num_layers):
        super().__init__()
        self.num_layers=num_layers
        self.vocab_size = vocab_size
        self.attention =  Attention(enc_hid_dim, dec_hid_dim)
        self.embedding = nn.Embedding(vocab_size, enc_hid_dim)
        self.rnn = nn.GRU(enc_hid_dim * 2+ enc_hid_dim, dec_hid_dim,num_layers = num_layers,dropout=0.3)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + enc_hid_dim , self.vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, dec_input, s, enc_output):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [seq_len*num_attrs, batch_size, enc_hid_dim * 2]

        dec_input = dec_input.unsqueeze(1) # dec_input = [batch_size, 1]
        dec_input =self.embedding(dec_input) # dec_input = [batch_size, 1] => [batch_size, 1,enc_hid_dim]

        dropout_dec_input = self.dropout(dec_input).transpose(0, 1) #  [batch_size, 1,enc_hid_dim]=>[1,batch_size,enc_hid_dim]

        # c = [1, batch_size, enc_hid_dim * 2] ；attention_probs = [batch_size , 1, seq_len*num_attrs]
        c,attention_probs = self.attention(s, enc_output)

        rnn_input = torch.cat((dropout_dec_input, c), dim = 2) # rnn_input = [1, batch_size, (enc_hid_dim * 2)+ enc_hid_dim]
        # dec_output=[1,batch_size,dec_hid_dim]  ; dec_hidden=[num_layers,batch_size,dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.repeat( self.num_layers,1,1))

        dec_output = dec_output.squeeze(0) # dec_output:[ batch_size, dec_hid_dim]

        c = c.squeeze(0)  # c:[batch_size, enc_hid_dim * 2]

        dropout_dec_input=dropout_dec_input.squeeze(0)  # dec_input:[batch_size, enc_hid_dim]

        pred = self.fc_out(torch.cat((dec_output, c, dropout_dec_input), dim = 1))# pred = [batch_size, output_dim]

        return pred, dec_hidden[-1].squeeze(0),attention_probs


class GRU_AE(nn.Module):
    def __init__(self, attribute_dims,enc_hidden_dim,encoder_num_layers,decoder_num_layers,dec_hidden_dim):
        super().__init__()
        encoders=[]
        decoders=[]
        self.attribute_dims=attribute_dims
        for i, dim in enumerate(attribute_dims):
            encoders.append( Encoder(int(dim + 1), enc_hidden_dim, encoder_num_layers, dec_hidden_dim))
            decoders.append( Decoder(int(dim + 1), enc_hidden_dim, dec_hidden_dim,decoder_num_layers))
        self.encoders=nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        # self.softmax = nn.LogSoftmax(dim=2)


    def forward(self, Xs):
        '''
        :param Xs:是多个属性，每一个属性作为一个X ：[batch_size, time_step]
        :return:
        '''
        trg_len = None
        outputs = [] #概率分布 probability map
        s = [] #解码层GRU初始隐藏表示
        enc_output = None
        for i, dim in enumerate(self.attribute_dims):
            X = Xs[i].transpose(0, 1)
            batch_size = X.shape[1]
            trg_len = X.shape[0]
            output_dim = int(dim)+1
            # 存储所有时刻的输出
            outputs.append(torch.zeros(trg_len, batch_size, output_dim).to(device))  # 存储decoder的所有输出

            enc_output_, s_ = self.encoders[i](X)

            if enc_output is None:
                enc_output = enc_output_
            else:
                enc_output = torch.cat((enc_output, enc_output_), dim=0)
            # enc_output = [trg_len*len(self.attribute_dims), batch_size, enc_hid_dim * 2]
            s.append(s_)

            outputs[i][0, :, X[0, 0]] = 1

        for i, dim in enumerate(self.attribute_dims):
            X = Xs[i].transpose(0, 1)
            s0 = s[i]
            dec_input= X[0, :] # target的第一列，即全是起始字符 teacher_forcing

            for t in range(1, trg_len):
                dec_output, s0 , _ = self.decoders[i](dec_input, s0, enc_output)

                # 存储每个时刻的输出
                outputs[i][t] = dec_output

                dec_input = X[t] #teacher_forcing

            outputs[i] = outputs[i].transpose(0, 1)

        return outputs



    def get_attention_probs(self, Xs):
        '''
        :param Xs:是多个属性，每一个属性作为一个X ：[batch_size, time_step]
        :return:
        '''
        trg_len = None
        attentions = []
        s = []
        enc_output = None
        for i, dim in enumerate(self.attribute_dims):
            X = Xs[i].transpose(0, 1)
            batch_size = X.shape[1]
            trg_len = X.shape[0]
            # 存储所有时刻的输出
            attentions.append(torch.zeros(trg_len, batch_size, trg_len,len(self.attribute_dims)).to(device))  # 存储attention的所有输出

            enc_output_, s_ = self.encoders[i](X)

            if enc_output is None:
                enc_output = enc_output_
            else:
                enc_output = torch.cat((enc_output, enc_output_), dim=0)
            # enc_output = [trg_len*len(self.attribute_dims), batch_size, enc_hid_dim * 2]
            s.append(s_)


        for i, dim in enumerate(self.attribute_dims):
            X = Xs[i].transpose(0, 1)
            batch_size = X.shape[1]
            s0 = s[i]
            dec_input= X[0, :] # target的第一列，即全是起始字符

            for t in range(1, trg_len):
                dec_output, s0 , attention_probs = self.decoders[i](dec_input, s0, enc_output)

                # 存储每个时刻的输出
                attentions[i][t] = attention_probs.squeeze(1).reshape((batch_size, trg_len,len(self.attribute_dims)))


                dec_input = X[t]
            attentions[i] = attentions[i].transpose(0, 1)

        return attentions
