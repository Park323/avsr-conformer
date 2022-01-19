import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from las_model.ksponspeech import KsponSpeechVocabulary
from las_model.layers import DotProductAttention, MultiHeadAttention

class Speller(nn.Module):

    def __init__(self, opt, vocab):
        super(Speller, self).__init__()

        self.num_classes=len(vocab)                              # number of classfication
        self.max_length=opt.las_model['decoder_max_length']                # a maximum allowed length for the sequence to be processed
        self.hidden_dim=opt.las_model['decoder_hidden_dim']                # dimension of RNN`s hidden state vector
        self.embedding_dim=opt.las_model['decoder_embedding_dim']
        self.num_layers=opt.las_model['decoder_num_layers']                # number of RNN layers
        self.rnn_type=opt.las_model['decoder_rnn_type']                    # type of RNN cell
        self.dropout_p=opt.las_model['decoder_dropout_p']                  # dropout probability
        self.pad_id=vocab.pad_id                                 # pad token`s id
        self.sos_id=vocab.sos_id                                 # start of sentence token`s id
        self.eos_id=vocab.eos_id                                 # end of sentence token`s id
        self.bidirectional=opt.las_model['decoder_bidirectional']
        self.num_heads = 4
        self.rnn_unit = getattr(nn, self.rnn_type.upper())

        if self.bidirectional:
            self.embedding_dim = self.embedding_dim*2

        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_p,
            batch_first=True
        )

        self.emb = nn.Embedding(self.num_classes, self.embedding_dim)
        #self.attention = DotProductAttention()
        self.attention = MultiHeadAttention(self.hidden_dim, self.num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim + self.hidden_dim,
                      self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.num_classes))

        self.softmax = nn.LogSoftmax(dim=-1)

    def init_hidden(self, batch_size):
        """
        initial first hidden layers in batch

        :param batch_size:
        :return:
        """
        hidden = Variable(torch.zeros(self.num_layers*2 if self.bidirectional else self.num_layers, batch_size, self.hidden_dim))
        cell = Variable(torch.zeros(self.num_layers*2 if self.bidirectional else self.num_layers, batch_size, self.hidden_dim))
        return (hidden.cuda(), cell.cuda())

    def forward_step(self, inputs, hc, encoder_outputs):
        """

        :param inputs:
        :param hc:
        :param encoder_outputs:
        :return:
        """
        decoder_output, hc = self.rnn(inputs, hc)
        #print(decoder_output.shape, encoder_outputs.shape)
        att_out, context = self.attention(decoder_output, encoder_outputs)
        #att_out = att_out.squeeze(dim=1)
        #print('att_out', att_out.shape, decoder_output.shape)
        mlp_input = torch.cat((decoder_output, att_out), dim=2)
        #print('mlp_input', mlp_input.shape)
        logit = self.softmax(self.mlp(mlp_input))

        return logit, hc, context

    def beam_search(self, inputs, hc, encoder_outputs, max_step, beam_size):
        btz = encoder_outputs.size(0)
        y_hats = torch.zeros(btz, max_step).long().cuda()
        logit, hc, context = self.forward_step(inputs, hc, encoder_outputs)
        output_words = logit.topk(beam_size)[1].squeeze(1)
        for bi in range(btz):
            b_output_words = output_words[bi, :].unsqueeze(0).transpose(1, 0).contiguous()
            b_inputs = self.emb(b_output_words)
            b_listener_features = encoder_outputs[bi, :, :].unsqueeze(0).expand((beam_size, -1, -1)).contiguous()
            if isinstance(hc, tuple):
                b_h = hc[0][:, bi, :].unsqueeze(1).expand((-1, beam_size, -1)).contiguous()
                b_c = hc[1][:, bi, :].unsqueeze(1).expand((-1, beam_size, -1)).contiguous()
                b_hc = (b_h, b_c)
            else:
                b_hc = hc[:, bi, :].unsqueeze(1).expand((-1, beam_size, -1)).contiguous()

            scores = torch.zeros(beam_size, 1).cuda()
            ids = torch.zeros(beam_size, max_step, 1).long().cuda()
            for step in range(max_step):
                logit, b_hc, context = self.forward_step(b_inputs, b_hc, b_listener_features)
                score, id = logit.topk(1)
                scores += score.squeeze(1)
                ids[:, step, :] = id.squeeze(1)
                output_word = logit.topk(1)[1].squeeze(-1)
                b_inputs = self.emb(output_word)
            # print(scores.squeeze(1).topk(1)[1])
            y_hats[bi, :] = ids[scores.squeeze(1).topk(1)[1], :].squeeze(2)
        return y_hats

    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.9, use_beam=False, beam_size=3):
        if targets is None:
            teacher_forcing_ratio = 0
        teacher_forcing = True if np.random.random_sample() < teacher_forcing_ratio else False

        if (targets is None) and (not teacher_forcing):
            max_step = self.max_length
        else:
            max_step = targets.size(1) - 1

        input_word = torch.zeros(encoder_outputs.size(0), 1).long().cuda()
        input_word[:, 0] = self.sos_id

        inputs = self.emb(input_word)
        hc = self.init_hidden(input_word.size(0))
        logits = []
        if use_beam:
            logits = self.beam_search(inputs, hc, encoder_outputs, max_step, beam_size)
        else:
            for step in range(max_step):
                logit, hc, context = self.forward_step(inputs, hc, encoder_outputs)
                logits.append(logit.squeeze())
                if teacher_forcing:
                    output_word = targets[:, step + 1:step + 2]
                else:
                    output_word = logit.topk(1)[1].squeeze(-1)
                inputs = self.emb(output_word)

            logits = torch.stack(logits, dim=1)
            # y_hats = torch.max(logits, dim=-1)[1]
        return logits

