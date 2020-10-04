# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from .mc_model import MCmodel
from .transformer import TransformerEncoder
from .wordrep import WordRep


class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.use_crf = data.use_crf
        print("build network...")

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss

        self.mcmodel = MCmodel(data)

        self.nsamples = data.HP_nsamples
        self.threshold = data.HP_threshold

        self.wordrep = WordRep(data)


        self.label_embedding = nn.Embedding(data.label_alphabet_size, data.HP_label_embed_dim)
        self.label_embedding.weight.data.copy_(torch.from_numpy(
            random_embedding_label(data.label_alphabet_size, data.HP_label_embed_dim, data.HP_label_embedding_scale)))

        self.word2hidden = nn.Linear(self.wordrep.total_size, data.d_model)
        self.label2hidden = nn.Linear(data.HP_label_embed_dim, data.d_model)

        self.encoder = TransformerEncoder(data.HP_model2_layer, data.d_model, data.HP_nhead,
                                          data.HP_dim_feedforward,dropout=data.HP_model2_dropout, dropout_attn=data.HP_attention_dropout,
                                          )

        self.model2_fc_dropout = nn.Dropout(data.HP_model2_dropout)

        self.hidden2tag = nn.Linear(data.d_model * 2, data.label_alphabet_size)
        self.m2_params = [self.wordrep, self.word2hidden, self.label2hidden, self.encoder, self.hidden2tag, self.label_embedding]

        if self.gpu:
            for i in range(len(self.m2_params)):
                self.m2_params[i] = self.m2_params[i].cuda()

    def neg_log_likelihood_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                char_seq_recover, batch_label, mask):
        batch_size, seq_len = word_inputs.size()
        mask = mask.eq(1)

        p, lstm_out, outs1, _ = self.mcmodel(word_inputs, feature_inputs, word_seq_lengths, char_inputs,
                                             char_seq_lengths,
                                             char_seq_recover)

        model1_preds = self.decode_seq(outs1, mask)

        uncertainty = epistemic_uncertainty(p, mask)
        label_mask = generate_label_mask(uncertainty, mask, threshold=self.threshold)

        # stage 2: forward model2 to get final labels
        model2_input_label_embed = torch.einsum("bsc,cd->bsd", [p.detach(), self.label_embedding.weight]).masked_fill(~mask.unsqueeze(-1), 0)

        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                      char_seq_recover)
        word_represent = self.word2hidden(word_represent)
        model2_input_label_embed = self.label2hidden(model2_input_label_embed)

        hh, hl = self.encoder(word_represent, model2_input_label_embed, mask)

        outs2 = self.hidden2tag(self.model2_fc_dropout(torch.cat([hh, hl], -1)))

        model2_preds = self.decode_seq(outs2, mask)

        predicted_seq = model1_preds.masked_fill(label_mask, 0) + model2_preds.masked_fill(~label_mask, 0)

        loss1 = self.get_loss(outs1, mask, batch_label)
        loss2 = self.get_loss(outs2, mask, batch_label)
        loss = loss1 + loss2

        if self.average_batch:
            loss = loss / batch_size

        return loss, predicted_seq

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                mask):
        mask = mask.eq(1)

        p, lstm_out, outs1, _ = self.mcmodel.MC_sampling(word_inputs, feature_inputs, word_seq_lengths,
                                                                      char_inputs,
                                                                      char_seq_lengths, char_seq_recover, self.nsamples)

        model1_preds = self.decode_seq(outs1, mask)
        uncertainty = epistemic_uncertainty(p, mask)
        label_mask = generate_label_mask(uncertainty, mask, threshold=self.threshold)

        # stage 2: forward model2 to get final labels
        model2_input_label_embed = torch.einsum("bsc,cd->bsd", [p, self.label_embedding.weight]).masked_fill(~mask.unsqueeze(-1), 0)

        word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                      char_seq_recover)
        word_represent = self.word2hidden(word_represent)
        model2_input_label_embed = self.label2hidden(model2_input_label_embed)

        hh, hl = self.encoder(word_represent, model2_input_label_embed, mask)
        outs2 = self.hidden2tag(self.model2_fc_dropout(torch.cat([hh, hl], -1)))
        model2_preds = self.decode_seq(outs2, mask)

        predicted_seq = model1_preds.masked_fill(label_mask, 0) + model2_preds.masked_fill(~label_mask, 0)

        return predicted_seq

    def decode_seq(self, outs, mask):
        preds = outs.argmax(-1)
        preds = preds.masked_fill(~mask, 0)  # mask padding words
        return preds

    def get_loss(self, outs, mask, batch_label):
        batch_size, seq_len = outs.size()[:2]

        loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
        loss = loss_function(outs.view(batch_size * seq_len, -1), batch_label.view(batch_size * seq_len))
        return loss

    def get_m2_params(self):
        return nn.ModuleList(self.m2_params).parameters()

def random_embedding_label(vocab_size, embedding_dim, scale):
    pretrain_emb = np.empty([vocab_size, embedding_dim])
    for index in range(vocab_size):
        pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
    return pretrain_emb


def epistemic_uncertainty(p, mask):
    '''
    calculate epistemic_uncertainty
    :param p: (batch, max_seq_len, num_labels)
    :param mask: (batch,max_seq_len) 1 means the position is valid (not masked).
    :return:  (batch, max_seq_len)
    '''
    hp = -((p + 1e-30) * (p + 1e-30).log()).sum(-1)
    hp = hp.masked_fill(mask == 0, 0)
    return hp


def generate_label_mask(hp, mask, topk=None, threshold=None):
    assert topk is not None or threshold is not None, "Must set topk or threshold!"
    if topk is not None:
        label_mask = hp.new_zeros(*hp.size()) == 1  # convert to BoolTensor(pytorch version >=1.2.0) or ByteTensor
        label_mask[..., :topk] = 1
        index = hp.sort(descending=True)[1]
        recover_index = index.sort()[1]
        label_mask = label_mask.gather(1, recover_index)
        return label_mask.masked_fill(~mask, 0)
    else:
        return (hp > threshold).masked_fill(~mask, 0)

