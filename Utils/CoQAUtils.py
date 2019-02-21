# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import os
import sys
import random
import string
import logging
import argparse
import unicodedata
from shutil import copyfile
from datetime import datetime
from collections import Counter
from collections import defaultdict
import torch
import msgpack
import json
import numpy as np
import pandas as pd
from Models.Bert.tokenization import BertTokenizer
from Utils.GeneralUtils import normalize_text, nlp
from Utils.Constants import *
from torch.autograd import Variable

POS = {w: i for i, w in enumerate([''] + list(nlp.tagger.labels))}
ENT = {w: i for i, w in enumerate([''] + nlp.entity.move_names)}

def build_embedding(embed_file, targ_vocab, wv_dim):
    vocab_size = len(targ_vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[0] = 0 # <PAD> should be all 0 (using broadcast)

    w2id = {w: i for i, w in enumerate(targ_vocab)}
    lineCnt = 0
    with open(embed_file, encoding="utf8") as f:
        for line in f:
            lineCnt = lineCnt + 1
            if lineCnt % 100000 == 0:
                print('.', end = '',flush=True)
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

def token2id_sent(sent, w2id, unk_id=None, to_lower=False):
    if to_lower:
        sent = sent.lower()
    w2id_len = len(w2id)    
    ids = [w2id[w] if w in w2id else unk_id for w in sent]
    return ids

def char2id_sent(sent, c2id, unk_id=None, to_lower=False):
    if to_lower:
        sent = sent.lower()
    cids = [[c2id["<STA>"]] + [c2id[c] if c in c2id else unk_id for c in w] + [c2id["<END>"]] for w in sent]
    return cids

def token2id(w, vocab, unk_id=None):
    return vocab[w] if w in vocab else unk_id

'''
 Generate feature per context word according to its exact match with question words
'''
def feature_gen(context, question):
    counter_ = Counter(w.text.lower() for w in context)
    total = sum(counter_.values())
    term_freq = [counter_[w.text.lower()] / total for w in context]
    question_word = {w.text for w in question}
    question_lower = {w.text.lower() for w in question}
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
    match_origin = [w.text in question_word for w in context]
    match_lower = [w.text.lower() in question_lower for w in context]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in context]
    C_features = list(zip(term_freq, match_origin, match_lower, match_lemma))
    return C_features    

'''
 Get upper triangle matrix from start and end scores (batch)
 Input:
  score_s: batch x context_len
  score_e: batch x context_len
  context_len: number of words in context
  max_len: maximum span of answer
  use_cuda: whether GPU is used
 Output:
  expand_score: batch x (context_len * context_len) 
'''
def gen_upper_triangle(score_s, score_e, max_len, use_cuda):
    batch_size = score_s.shape[0]
    context_len = score_s.shape[1]
    # batch x context_len x context_len
    expand_score = score_s.unsqueeze(2).expand([batch_size, context_len, context_len]) +\
        score_e.unsqueeze(1).expand([batch_size, context_len, context_len])
    score_mask = torch.ones(context_len)
    if use_cuda:
        score_mask = score_mask.cuda()
    score_mask = torch.ger(score_mask, score_mask).triu().tril(max_len - 1)
    empty_mask = score_mask.eq(0).unsqueeze(0).expand_as(expand_score)
    expand_score.data.masked_fill_(empty_mask.data, -float('inf'))
    return expand_score.contiguous().view(batch_size, -1) # batch x (context_len * context_len)    

class BatchGen:
    def __init__(self, opt, data, use_cuda, vocab, char_vocab, evaluation=False):
        # file_name = os.path.join(self.spacyDir, 'coqa-' + dataset_label + '-preprocessed.json')

        self.data = data
        self.use_cuda = use_cuda 
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.evaluation = evaluation
        self.opt = opt
        if 'PREV_ANS' in self.opt:
            self.prev_ans = self.opt['PREV_ANS']
        else:
            self.prev_ans = 2

        if 'PREV_QUES' in self.opt:
            self.prev_ques = self.opt['PREV_QUES']
        else:
            self.prev_ques = 0

        self.use_char_cnn = 'CHAR_CNN' in self.opt

        self.bert_tokenizer = None
        if 'BERT' in self.opt:
            if 'BERT_LARGE' in opt:
                print('Using BERT Large model')
                tokenizer_file = os.path.join(opt['datadir'], opt['BERT_large_tokenizer_file'])
                print('Loading tokenizer from', tokenizer_file)
                self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_file)
            else:
                print('Using BERT base model')
                tokenizer_file = os.path.join(opt['datadir'], opt['BERT_tokenizer_file'])
                print('Loading tokenizer from', tokenizer_file)
                self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_file)

        self.answer_span_in_context = 'ANSWER_SPAN_IN_CONTEXT_FEATURE' in self.opt

        self.ques_max_len = (30 + 1) * self.prev_ans + (25 + 1) * (self.prev_ques + 1)
        self.char_max_len = 30

        print('*****************')
        print('prev_ques   :', self.prev_ques)
        print('prev_ans    :', self.prev_ans)
        print('ques_max_len:', self.ques_max_len)
        print('*****************')

        c2id = {c: i for i, c in enumerate(char_vocab)}
        
        # random shuffle for training
        if not evaluation:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def bertify(self, words):
        if self.bert_tokenizer is None:
            return None

        bpe = ['[CLS]']
        x_bert_offsets = []
        for word in words:
            now = self.bert_tokenizer.tokenize(word)
            x_bert_offsets.append([len(bpe), len(bpe) + len(now)])
            bpe.extend(now)
        
        bpe.append('[SEP]')

        x_bert = self.bert_tokenizer.convert_tokens_to_ids(bpe)
        return x_bert, x_bert_offsets

    def __iter__(self):
        data = self.data
        MAX_ANS_SPAN = 15
        for datum in data:
            if not self.evaluation:
                # remove super long answers for training
                datum['qas'] = [qa for qa in datum['qas'] if len(qa['annotated_answer']['word']) == 1 or qa['answer_span'][1] - qa['answer_span'][0] < MAX_ANS_SPAN]
            
            if len(datum['qas']) == 0:
                continue

            context_len = len(datum['annotated_context']['wordid'])
            x_len = context_len

            qa_len = len(datum['qas'])

            batch_size = qa_len
            x = torch.LongTensor(1, x_len).fill_(0)
            x_char = torch.LongTensor(1, x_len, self.char_max_len).fill_(0)
            if 'BERT' in self.opt:
                x_bert, x_bert_offsets = self.bertify(datum['annotated_context']['word'])
                x_bert_mask = torch.LongTensor(1, len(x_bert)).fill_(1)
                x_bert = torch.tensor([x_bert], dtype = torch.long)
                x_bert_offsets = torch.tensor([x_bert_offsets], dtype = torch.long)

            x_pos = torch.LongTensor(1, x_len).fill_(0)
            x_ent = torch.LongTensor(1, x_len).fill_(0)            
            
            if self.answer_span_in_context:
                x_features = torch.Tensor(batch_size, x_len, 5).fill_(0)
            else:
                x_features = torch.Tensor(batch_size, x_len, 4).fill_(0)

            query = torch.LongTensor(batch_size, self.ques_max_len).fill_(0)
            query_char = torch.LongTensor(batch_size, self.ques_max_len, self.char_max_len).fill_(0)
            query_bert_offsets = torch.LongTensor(batch_size, self.ques_max_len, 2).fill_(0)
            q_bert_list = []
            ground_truth = torch.LongTensor(batch_size, 2).fill_(-1)

            context_id = datum['id']
            context_str = datum['context']
            context_words = datum['annotated_context']['word']
            context_word_offsets = datum['raw_context_offsets']
            answer_strs = []
            turn_ids = []

            x[0, :context_len] = torch.LongTensor(datum['annotated_context']['wordid'])
            if self.use_char_cnn:
                for j in range(context_len):
                    t = min(len(datum['annotated_context']['charid'][j]), self.char_max_len)
                    x_char[0, j, :t] = torch.LongTensor(datum['annotated_context']['charid'][j][:t])

            x_pos[0, :context_len] = torch.LongTensor(datum['annotated_context']['pos_id'])
            x_ent[0, :context_len] = torch.LongTensor(datum['annotated_context']['ent_id'])

            for i in range(qa_len):
                x_features[i, :context_len, :4] = torch.Tensor(datum['qas'][i]['context_features'])
                turn_ids.append(int(datum['qas'][i]['turn_id']))
                # query
                p = 0

                ques_words = []
                # put in qa
                for j in range(i - self.prev_ans, i + 1):
                    if j < 0:
                        continue;
                    if not self.evaluation and datum['qas'][j]['answer_span'][0] == -1: # questions with "unknown" answers are filtered out
                        continue    

                    q = [2] + datum['qas'][j]['annotated_question']['wordid']
                    q_char = [[0]] + datum['qas'][j]['annotated_question']['charid']
                    if j >= i - self.prev_ques and p + len(q) <= self.ques_max_len:
                        ques_words.extend(['<Q>'] + datum['qas'][j]['annotated_question']['word'])
                        # <Q>: 2, <A>: 3                    
                        query[i, p:(p+len(q))] = torch.LongTensor(q)
                        if self.use_char_cnn:
                            for k in range(len(q_char)):
                                t = min(self.char_max_len, len(q_char[k]))
                                query_char[i, p + k, :t] = torch.LongTensor(q_char[k][:t])
                        ques = datum['qas'][j]['question'].lower()
                        p += len(q)

                    a = [3] + datum['qas'][j]['annotated_answer']['wordid']
                    a_char = [[0]] + datum['qas'][j]['annotated_answer']['charid']
                    if j < i and j >= i - self.prev_ans and p + len(a) <= self.ques_max_len:
                        ques_words.extend(['<A>'] + datum['qas'][j]['annotated_answer']['word'])
                        query[i, p:(p+len(a))] = torch.LongTensor(a) 
                        if self.use_char_cnn:
                            for k in range(len(a_char)):
                                t = min(self.char_max_len, len(a_char[k]))
                                query_char[i, p + k, :t] = torch.LongTensor(a_char[k][:t])
                        p += len(a)

                        if self.answer_span_in_context:
                            st = datum['qas'][j]['answer_span'][0]
                            ed = datum['qas'][j]['answer_span'][1] + 1
                            x_features[i, st:ed, 4] = 1.0

                if 'BERT' in self.opt:
                    now_bert, now_bert_offsets = self.bertify(ques_words)
                    query_bert_offsets[i, :len(now_bert_offsets), :] = torch.tensor(now_bert_offsets, dtype = torch.long)
                    q_bert_list.append(now_bert)

                # answer
                ground_truth[i, 0] = datum['qas'][i]['answer_span'][0]
                ground_truth[i, 1] = datum['qas'][i]['answer_span'][1]
                answer = datum['qas'][i]['raw_answer']

                if answer.lower() in ['yes', 'yes.']:
                    ground_truth[i, 0] = -1
                    ground_truth[i, 1] = 0
                    answer_str = 'yes'

                if answer.lower() in ['no', 'no.']:
                    ground_truth[i, 0] = 0
                    ground_truth[i, 1] = -1
                    answer_str = 'no'

                if answer.lower() == ['unknown', 'unknown.']:
                    ground_truth[i, 0] = -1
                    ground_truth[i, 1] = -1
                    answer_str = 'unknown'

                if ground_truth[i, 0] >= 0 and ground_truth[i, 1] >= 0:
                    answer_str = answer
                
                all_viable_answers = [answer_str]
                if 'additional_answers' in datum['qas'][i]:
                    all_viable_answers.extend(datum['qas'][i]['additional_answers'])
                answer_strs.append(all_viable_answers)


            if 'BERT' in self.opt:
                bert_len = max([len(s) for s in q_bert_list])
                query_bert = torch.LongTensor(batch_size, bert_len).fill_(0)
                query_bert_mask = torch.LongTensor(batch_size, bert_len).fill_(0)
                for i in range(len(q_bert_list)):
                    query_bert[i, :len(q_bert_list[i])] = torch.LongTensor(q_bert_list[i])
                    query_bert_mask[i, :len(q_bert_list[i])] = 1
                if self.use_cuda:
                    x_bert = Variable(x_bert.cuda(async=True))
                    x_bert_mask = Variable(x_bert_mask.cuda(async=True))
                    query_bert = Variable(query_bert.cuda(async=True))
                    query_bert_mask = Variable(query_bert_mask.cuda(async=True))
                else:
                    x_bert = Variable(x_bert)
                    x_bert_mask = Variable(x_bert_mask)
                    query_bert = Variable(query_bert)
                    query_bert_mask = Variable(query_bert_mask)   
            else:
                x_bert = None
                x_bert_mask = None
                x_bert_offsets = None
                query_bert = None        
                query_bert_mask = None
                query_bert_offsets = None

            if self.use_char_cnn:
                x_char_mask = 1 - torch.eq(x_char, 0)
                query_char_mask = 1 - torch.eq(query_char, 0)
                if self.use_cuda:
                    x_char = Variable(x_char.cuda(async=True))
                    x_char_mask = Variable(x_char_mask.cuda(async=True))
                    query_char = Variable(query_char.cuda(async=True))
                    query_char_mask = Variable(query_char_mask.cuda(async=True))
                else:
                    x_char = Variable(x_char)
                    x_char_mask = Variable(x_char_mask)
                    query_char = Variable(query_char)
                    query_char_mask = Variable(query_char_mask)
            else:
                x_char = None
                x_char_mask = None
                query_char = None                               
                query_char_mask = None                               

            x_mask = 1 - torch.eq(x, 0)
            query_mask = 1 - torch.eq(query, 0)
            if self.use_cuda:
                x = Variable(x.cuda(async=True))
                x_mask = Variable(x_mask.cuda(async=True))                
                x_features = Variable(x_features.cuda(async=True))
                x_pos = Variable(x_pos.cuda(async=True))
                x_ent = Variable(x_ent.cuda(async=True))
                query = Variable(query.cuda(async=True))
                query_mask = Variable(query_mask.cuda(async=True))                
                ground_truth = Variable(ground_truth.cuda(async=True))
            else:
                x = Variable(x)
                x_mask = Variable(x_mask)                
                x_features = Variable(x_features)
                x_pos = Variable(x_pos)
                x_ent = Variable(x_ent)
                query = Variable(query)
                query_mask = Variable(query_mask)
                ground_truth = Variable(ground_truth)
            yield(x, x_mask, x_char, x_char_mask, x_features, x_pos, x_ent, x_bert, x_bert_mask, x_bert_offsets, query, query_mask, query_char, query_char_mask,
            query_bert, query_bert_mask, query_bert_offsets, ground_truth, context_str, context_words, context_word_offsets, answer_strs, context_id, turn_ids)

#===========================================================================
#=================== For standard evaluation in CoQA =======================
#===========================================================================

def ensemble_predict(pred_list, score_list, voteByCnt = False):
    predictions, best_scores = [], []
    pred_by_examples = list(zip(*pred_list))
    score_by_examples = list(zip(*score_list))
    for phrases, scores in zip(pred_by_examples, score_by_examples):
        d = defaultdict(float)
        firstappear = defaultdict(int)
        for phrase, phrase_score, index in zip(phrases, scores, range(len(scores))):
            d[phrase] += 1. if voteByCnt else phrase_score
            if not phrase in firstappear:
                firstappear[phrase] = -index
        predictions += [max(d.items(), key=lambda pair: (pair[1], firstappear[pair[0]]))[0]]
        best_scores += [max(d.items(), key=lambda pair: (pair[1], firstappear[pair[0]]))[1]]
    return (predictions, best_scores)

def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0

    if len(answers) == 0:
        return 1. if len(pred) == 0 else 0.
    
    g_tokens = _normalize_answer(pred).split()
    ans_tokens = [_normalize_answer(answer).split() for answer in answers]
    scores = [_score(g_tokens, a) for a in ans_tokens]
    if len(ans_tokens) == 1:
        score = scores[0]
    else:
        score = 0
        for i in range(len(ans_tokens)):
            scores_one_out = scores[:i] + scores[(i + 1):]
            score += max(scores_one_out)
        score /= len(ans_tokens)
    return score

def score(pred, truth, final_json):
    assert len(pred) == len(truth)
    no_ans_total = no_total = yes_total = normal_total = total = 0
    no_ans_f1 = no_f1 = yes_f1 = normal_f1 = f1 = 0
    all_f1s = []
    for p, t, j in zip(pred, truth, final_json):
        total += 1
        this_f1 = _f1_score(p, t)
        f1 += this_f1
        all_f1s.append(this_f1)
        if t[0].lower() == 'no':
            no_total += 1
            no_f1 += this_f1
        elif t[0].lower() == 'yes':
            yes_total += 1
            yes_f1 += this_f1
        elif t[0].lower() == 'unknown':
            no_ans_total += 1
            no_ans_f1 += this_f1
        else:
            normal_total += 1
            normal_f1 += this_f1

    f1 = 100. * f1 / total
    if no_total == 0:
        no_f1 = 0.
    else:
        no_f1 = 100. * no_f1 / no_total
    if yes_total == 0:
        yes_f1 = 0
    else:
        yes_f1 = 100. * yes_f1 / yes_total
    if no_ans_total == 0:
        no_ans_f1 = 0.
    else:
        no_ans_f1 = 100. * no_ans_f1 / no_ans_total
    normal_f1 = 100. * normal_f1 / normal_total
    result = {
        'total': total,
        'f1': f1,
        'no_total': no_total,
        'no_f1': no_f1,
        'yes_total': yes_total,
        'yes_f1': yes_f1,
        'no_ans_total': no_ans_total,
        'no_ans_f1': no_ans_f1,
        'normal_total': normal_total,
        'normal_f1': normal_f1,
    }
    return result, all_f1s

def score_each_instance(pred, truth):
    assert len(pred) == len(truth)
    total = 0
    f1_scores = []
    for p, t in zip(pred, truth):
        total += 1
        f1_scores.append(_f1_score(p, t))
    f1_scores = [100. * x / total for x in f1_scores]
    return f1_scores

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
