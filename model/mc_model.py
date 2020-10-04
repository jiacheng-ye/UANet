# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep


class MCmodel(nn.Module):
    def __init__(self, data):
        super(MCmodel, self).__init__()
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.model1_fc_dropout = data.HP_model1_dropout
        self.model1_in_dropout = data.HP_bayesian_lstm_dropout[0]
        self.bilstm_flag = data.HP_bilstm
        self.hidden_dim = data.HP_hidden_dim
        # word embedding
        self.wordrep = WordRep(data)

        self.input_size = self.wordrep.total_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.lstms = nn.ModuleList([nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True,
                                          bidirectional=self.bilstm_flag)])
        for _ in range(data.HP_model1_layer-1):
            self.lstms.append(nn.LSTM(data.HP_hidden_dim, lstm_hidden, num_layers=1, batch_first=True,
                                          bidirectional=self.bilstm_flag))

        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)

        if self.gpu:
            self.lstms = self.lstms.cuda()
            self.hidden2tag = self.hidden2tag.cuda()

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        word_represent = self.forward_word(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                           char_seq_recover)
        return self.forward_rest(word_represent, word_seq_lengths)

    def forward_word(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                     char_seq_recover):
        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                      char_seq_recover)
        return word_represent

    def forward_rest(self, word_represent, word_seq_lengths):
        if not self.training:
            ordered_lens, index = word_seq_lengths.sort(descending=True)
            ordered_x = word_represent[index]
        else:
            ordered_x, ordered_lens = word_represent, word_seq_lengths

        for i,lstm in enumerate(self.lstms):
            ordered_x = add_dropout(ordered_x, self.model1_in_dropout)
            pack_input = pack_padded_sequence(ordered_x, ordered_lens, batch_first=True)
            pack_output, _ = lstm(pack_input)
            ordered_x, _ = pad_packed_sequence(pack_output, batch_first=True)

        if not self.training:
            recover_index = index.argsort()
            lstm_out = ordered_x[recover_index]
        else:
            lstm_out = ordered_x

        h2t_in = add_dropout(lstm_out, self.model1_fc_dropout)
        outs = self.hidden2tag(h2t_in)

        p = F.softmax(outs, -1)
        return p, lstm_out, outs, word_represent

    def MC_sampling(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                    char_seq_recover, mc_steps):
        '''

        :param word_inputs: (batch, max_seq_len)
        :param char_ids: (batch, max_seq_len, max_word_len)
        :param lens: (batch)
        :param mc_steps: scalar
        :return:
        '''

        word_represent = self.forward_word(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                           char_seq_recover)
        batch, max_seq_len = word_represent.size()[:2]

        _word_represent = word_represent.repeat([mc_steps] + [1 for _ in range(1, len(word_represent.size()))])
        _word_seq_lengths = word_seq_lengths.repeat([mc_steps] + [1 for _ in range(1, len(word_seq_lengths.size()))])

        p, lstm_out, outs, _ = self.forward_rest(_word_represent, _word_seq_lengths)

        p = p.reshape(mc_steps, batch, max_seq_len, -1).mean(0)
        lstm_out = lstm_out.reshape(mc_steps, batch, max_seq_len, -1).mean(0)
        outs = outs.reshape(mc_steps, batch, max_seq_len, -1).mean(0)

        return p, lstm_out, outs, word_represent


def add_dropout(x, dropout):
    ''' x: batch * seq_len * hidden '''
    return F.dropout2d(x.transpose(1,2)[...,None], p=dropout, training=True).squeeze(-1).transpose(1,2)