import torch.nn as nn
import torch
import math

#originaly use pBLSTMlayer but it need timestamp%2==0 condition
#condition modifiation is very hard so use just lstm stack
class pBLSTMLayer(nn.Module):
    def __init__(
            self,
            input_feature,
            hidden_dim,
            bidirectional,
            rnn_unit='LSTM',
            dropout_p=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature * 2, hidden_dim, 1, bidirectional=bidirectional,
                                   dropout=dropout_p, batch_first=True)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        timestep = inputs.size(1)
        feature_dim = inputs.size(2)
        # Reduce time resolution need 2^3 so add zeros
        if timestep % 2:
            #print(batch_size, timestep, feature_dim)
            zeros = torch.zeros((inputs.size(0), 1, inputs.size(2))).cuda()
            inputs = torch.cat([inputs, zeros], dim=1)
            timestep += 1
        input_x = inputs.contiguous().view(batch_size, int(timestep / 2), feature_dim * 2)
        # Bidirectional RNN
        output, hidden = self.BLSTM(input_x)
        return output, hidden

class Listener(nn.Module):
    def __init__(self, opt):
        super(Listener, self).__init__()

        self.input_size=opt.audio['n_mels']                   # size of input
        self.hidden_dim=opt.las_model['encoder_hidden_dim']       # dimension of RNN`s hidden state
        self.dropout_p=opt.las_model['encoder_dropout_p']         # dropout probability
        self.num_layers=opt.las_model['num_encoder_layers']       # number of RNN layers
        self.bidirectional=opt.las_model['encoder_bidirectional'] # if True, becomes a bidirectional encoder
        self.rnn_type=opt.las_model['encoder_rnn_type']
        
        # 3pyramid BLSTM
        self.pLSTM_layer0 = pBLSTMLayer(self.input_size, self.hidden_dim, self.bidirectional, self.rnn_type)
        self.pLSTM_layers = nn.ModuleList()
        for i in range(1, self.num_layers):
            self.pLSTM_layers.append(pBLSTMLayer(self.hidden_dim * 2, self.hidden_dim, self.bidirectional, self.rnn_type, self.dropout_p))


    def forward(self, inputs, input_lengths):
        output, _ = self.pLSTM_layer0(inputs)
        for i in range(len(self.pLSTM_layers)):
            output, _ = self.pLSTM_layers[i](output)

        return output
