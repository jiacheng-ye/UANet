# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import sys
import argparse
import random
import torch
import gc
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from utils.data import Data
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils.optimizer import *
from model.seqmodel import SeqModel

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os


def data_initialization(data):
    # data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def recover_word(word_ids, mask_variable, word_alphabet, word_recover):
    word_ids = word_ids[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size, seq_len = word_ids.size()

    mask = mask_variable.cpu().data.numpy()
    word_ids = word_ids.cpu().data.numpy()

    word_texts = []
    for idx in range(batch_size):
        words = [word_alphabet.get_instance(word_ids[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        word_texts.append(words)
    return word_texts


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print("Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)

    pred_results = []
    gold_results = []

    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1

    model.eval()
    with torch.no_grad():
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = instances[start:end]
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
                instance, data.HP_gpu, True)

            tag_seq = model(batch_word, batch_features, batch_wordlen,
                                                                batch_char,
                                                                batch_charlen, batch_charrecover,
                                                                mask)
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
            pred_results += pred_label
            gold_results += gold_label

    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)

    if data.seg:
        score = f
        print("%s: time: %.2f s, speed: %.2f doc/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f; \n" %
              (name, decode_time, speed, acc, p, r, f))
    else:
        score = acc
        print("%s: time: %.2f s speed: %.2f doc/s; acc: %.4f; \n" % (name, decode_time, speed, acc))

    if name == 'raw':
        print("save predicted results to %s" % data.decode_dir)
        data.write_decoded_results(pred_results, name)

    return score, pred_results


def batchify_with_label(input_batch_list, gpu, if_train=False):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentenc. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir + "/data.dset"
    if data.save_model:
        data.save(save_data_name)

    batch_size = data.HP_batch_size
    train_num = len(data.train_Ids)
    total_batch = train_num // batch_size + 1

    model = SeqModel(data)
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print(model)
    print("pytorch total params: %d" % pytorch_total_params)

    ## model 1 optimizer
    lr_detail1 = [{"params": filter(lambda p: p.requires_grad, model.mcmodel.parameters()), "lr": data.HP_lr},
                  ]
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(lr_detail1,
                              momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(lr_detail1, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(lr_detail1, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(lr_detail1, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(lr_detail1, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s" % (data.optimizer))
        exit(1)

    ## model 2 optimizer
    optimizer2 = AdamW(model.get_m2_params(), lr=data.HP_lr2, weight_decay=data.HP_l2)
    t_total = total_batch * data.HP_iteration
    warmup_step = int(data.warmup_step * t_total)
    scheduler2 = WarmupLinearSchedule(optimizer2, warmup_step, t_total)

    best_dev = -10
    best_test = -10
    max_test = -10
    max_test_epoch = -1
    max_dev_epoch = -1

    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        print("\n ###### Epoch: %s/%s ######" % (idx, data.HP_iteration))  # print (self.train_Ids)
        if data.optimizer.lower() == "sgd":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)

        sample_loss = 0
        total_loss = 0
        random.shuffle(data.train_Ids)

        model.train()
        model.zero_grad()
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]

            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
                instance, data.HP_gpu, True)
            loss, tag_seqs = model.neg_log_likelihood_loss(batch_word, batch_features, batch_wordlen, batch_char,
                                                          batch_charlen, batch_charrecover, batch_label, mask)

            sample_loss += loss.item()
            total_loss += loss.item()
            if end % 500 == 0:
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0

            loss.backward()
            clip_grad_norm_(model.parameters(), data.clip_grad)

            optimizer.step()
            optimizer2.step()
            scheduler2.step()
            model.zero_grad()

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2f s, speed: %.2f doc/s,  total loss: %s" % (
            idx, epoch_cost, train_num / epoch_cost, total_loss))

        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)

        # dev
        dev_score, _ = evaluate(data, model, "dev")

        # test
        test_score, _ = evaluate(data, model, "test")

        if max_test < test_score:
            max_test_epoch = idx
        max_test = max(test_score, max_test)
        if dev_score > best_dev:
            print("Exceed previous best dev score")
            best_test = test_score
            best_dev = dev_score
            max_dev_epoch = idx
            if data.save_model:
                model_name = data.model_dir + "/best_model.ckpt"
                print("Save current best model in file:", model_name)
                torch.save(model.state_dict(), model_name)

        print("Score summary: max dev (%d): %.4f, test: %.4f; max test (%d): %.4f" % (
            max_dev_epoch, best_dev, best_test, max_test_epoch, max_test))

        gc.collect()


def load_model_decode(data):
    print("Load Model from dir: ", data.model_dir)
    model = SeqModel(data)
    model_name = data.model_dir + "/best_model.ckpt"
    model.load_state_dict(torch.load(model_name))

    evaluate(data, model, "raw")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with UANet')
    parser.add_argument('--config', help='Configuration File')

    parser.add_argument('--train_dir', default='data/conll2003/train.txt', help='train_file')
    parser.add_argument('--dev_dir', default='data/conll2003/dev.txt', help='dev_file')
    parser.add_argument('--test_dir', default='data/conll2003/test.txt', help='test_file')
    parser.add_argument('--raw_dir', default='data/conll2003/test.txt')
    parser.add_argument('--model_dir', default='outs', help='model_file')

    parser.add_argument('--seg', default=True)
    parser.add_argument('--save_model', default=True)

    parser.add_argument('--word_emb_dir', default=None, help='word_emb_dir')
    parser.add_argument('--norm_word_emb', default=False)
    parser.add_argument('--norm_char_emb', default=False)
    parser.add_argument('--number_normalized', default=True)

    # training setting
    parser.add_argument('--status', choices=['train', 'decode'], default='train')
    parser.add_argument('--iteration', default=100)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--ave_batch_loss', default=False)
    parser.add_argument('--seed', default=920)

    # word representation
    parser.add_argument('--use_char', default=True)
    parser.add_argument('--char_emb_dim', default=30)
    parser.add_argument('--char_seq_feature', choices=['CNN', 'GRU', 'LSTM'], default='CNN')
    parser.add_argument('--char_hidden_dim', default=50, help="number of filter")
    parser.add_argument('--word_emb_dim', default=100)
    parser.add_argument('--dropout', default=0.5)

    # model1 parameter
    parser.add_argument('--bayesian_lstm_dropout', default=0.01)
    parser.add_argument('--model1_dropout', default=0.25)
    parser.add_argument('--hidden_dim', default=400)
    parser.add_argument('--model1_layer', default=1)
    parser.add_argument('--bilstm', default=True)
    parser.add_argument('--nsample', default=8)

    parser.add_argument('--threshold', default=0.55)

    # model2 parameter
    parser.add_argument('--label_embed_dim', default=400)
    parser.add_argument('--label_embedding_scale', default=600)
    parser.add_argument('--model2_layer', default=2)
    parser.add_argument('--d_head', default=80)
    parser.add_argument('--n_head', default=7)
    parser.add_argument('--model2_dropout', default=0.2)
    parser.add_argument('--attention_dropout', default=0.15)

    parser.add_argument('--use_crf', default=False)

    # optimizer
    parser.add_argument('--clip_grad', default=5)
    parser.add_argument('--l2', default=1e-6)
    ## model1 optimizer
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--learning_rate', default=0.015)
    parser.add_argument('--lr_decay', default=0.05)
    parser.add_argument('--momentum', default=0.9)
    ## model2 optimizer
    parser.add_argument('--warmup_step', default=0.1)
    parser.add_argument('--learning_rate2', default=0.0001)

    args = parser.parse_args()

    seed_num = int(args.seed)
    print("Seed num:", seed_num)
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    torch.random.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

    data = Data()
    data.HP_gpu = torch.cuda.is_available()

    if args.status == 'train':
        print("MODE: train")
        data.read_config(args)

        import uuid

        uid = uuid.uuid4().hex[:6]
        data.model_dir = data.model_dir + "_" + uid
        print("model dir: %s" % uid)
        if not os.path.exists(data.model_dir):
            os.mkdir(data.model_dir)

        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)
        print("model dir: %s" % uid)
    elif args.status == 'decode':
        print("MODE: decode")
        data.load(args.model_dir + "/data.dset")
        data.read_config(args)
        data.show_data_summary()
        data.generate_instance('raw')

        load_model_decode(data)
