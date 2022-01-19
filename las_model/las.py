import torch
import torch.nn as nn
import torch.nn.functional as F
from las_model.encoder import Listener
from las_model.decoder import Speller

class ListenAttendSpell(nn.Module):
    def __init__(self, opt, vocab):
        super(ListenAttendSpell, self).__init__()
        self.listener = Listener(opt)
        self.speller = Speller(opt, vocab)

    def forward(self, dum1, dum2, inputs, inputs_length, targets, targets_length, teacher_forcing_ratio=0.9, use_beam=False, beam_size=3):
        # dum1, dum2 : video_inputs, video_length

        encoder_outputs = self.listener(inputs, inputs_length)
        decoder_outputs = self.speller(encoder_outputs, targets, teacher_forcing_ratio, use_beam, beam_size)
        return decoder_outputs

    def greedy_search(self, inputs, inputs_length):
        with torch.no_grad():
            outputs = self.forward(None, None, inputs, inputs_length, None, None, 0.0)

        return outputs.max(-1)[1]
