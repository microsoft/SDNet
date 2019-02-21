# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn.parameter import Parameter
from Models.Bert.Bert import Bert
from Models.Layers import MaxPooling, CNN, dropout, RNN_from_opt, set_dropout_prob, weighted_avg, set_seq_dropout, Attention, DeepAttention, LinearSelfAttn, GetFinalScores
from Utils.CoQAUtils import POS, ENT

'''
 SDNet
'''
class SDNet(nn.Module):
    def __init__(self, opt, word_embedding):
        super(SDNet, self).__init__()
        print('SDNet model\n')

        self.opt = opt
        self.use_cuda = (self.opt['cuda'] == True)
        set_dropout_prob(0.0 if not 'DROPOUT' in opt else float(opt['DROPOUT']))
        set_seq_dropout('VARIATIONAL_DROPOUT' in self.opt)

        x_input_size = 0
        ques_input_size = 0

        self.vocab_size = int(opt['vocab_size'])
        vocab_dim = int(opt['vocab_dim'])
        self.vocab_embed = nn.Embedding(self.vocab_size, vocab_dim, padding_idx = 1)
        self.vocab_embed.weight.data = word_embedding

        x_input_size += vocab_dim
        ques_input_size += vocab_dim

        if 'CHAR_CNN' in self.opt:
            print('CHAR_CNN')
            char_vocab_size = int(opt['char_vocab_size'])
            char_dim = int(opt['char_emb_size'])
            char_hidden_size = int(opt['char_hidden_size'])
            self.char_embed = nn.Embedding(char_vocab_size, char_dim, padding_idx = 1)
            self.char_cnn = CNN(char_dim, 3, char_hidden_size)
            self.maxpooling = MaxPooling()
            x_input_size += char_hidden_size
            ques_input_size += char_hidden_size

        if 'TUNE_PARTIAL' in self.opt:
            print('TUNE_PARTIAL')
            self.fixed_embedding = word_embedding[opt['tune_partial']:]
        else:    
            self.vocab_embed.weight.requires_grad = False

        cdim = 0
        self.use_contextual = False

        if 'BERT' in self.opt:
            print('Using BERT')
            self.Bert = Bert(self.opt)
            if 'LOCK_BERT' in self.opt:
                print('Lock BERT\'s weights')
                for p in self.Bert.parameters():
                    p.requires_grad = False
            if 'BERT_LARGE' in self.opt:
                print('BERT_LARGE')
                bert_dim = 1024
                bert_layers = 24
            else:
                bert_dim = 768
                bert_layers = 12

            print('BERT dim:', bert_dim, 'BERT_LAYERS:', bert_layers)    

            if 'BERT_LINEAR_COMBINE' in self.opt:
                print('BERT_LINEAR_COMBINE')
                self.alphaBERT = nn.Parameter(torch.Tensor(bert_layers), requires_grad=True)
                self.gammaBERT = nn.Parameter(torch.Tensor(1, 1), requires_grad=True)
                torch.nn.init.constant(self.alphaBERT, 1.0)
                torch.nn.init.constant(self.gammaBERT, 1.0)
                
            cdim = bert_dim
            x_input_size += bert_dim
            ques_input_size += bert_dim

        self.pre_align = Attention(vocab_dim, opt['prealign_hidden'], correlation_func = 3, do_similarity = True)
        x_input_size += vocab_dim

        pos_dim = opt['pos_dim']
        ent_dim = opt['ent_dim']
        self.pos_embedding = nn.Embedding(len(POS), pos_dim)
        self.ent_embedding = nn.Embedding(len(ENT), ent_dim)

        x_feat_len = 4
        if 'ANSWER_SPAN_IN_CONTEXT_FEATURE' in self.opt:
            print('ANSWER_SPAN_IN_CONTEXT_FEATURE')
            x_feat_len += 1

        x_input_size += pos_dim + ent_dim + x_feat_len

        print('Initially, the vector_sizes [doc, query] are', x_input_size, ques_input_size)

        addtional_feat = cdim if self.use_contextual else 0

        # RNN context encoder
        self.context_rnn, context_rnn_output_size = RNN_from_opt(x_input_size, opt['hidden_size'],
            num_layers=opt['in_rnn_layers'], concat_rnn=opt['concat_rnn'], add_feat=addtional_feat)
        # RNN question encoder
        self.ques_rnn, ques_rnn_output_size = RNN_from_opt(ques_input_size, opt['hidden_size'],
            num_layers=opt['in_rnn_layers'], concat_rnn=opt['concat_rnn'], add_feat=addtional_feat)

        # Output sizes of rnn encoders
        print('After Input LSTM, the vector_sizes [doc, query] are [', context_rnn_output_size, ques_rnn_output_size, '] *', opt['in_rnn_layers'])

        # Deep inter-attention
        self.deep_attn = DeepAttention(opt, abstr_list_cnt=opt['in_rnn_layers'], 
            deep_att_hidden_size_per_abstr=opt['deep_att_hidden_size_per_abstr'], correlation_func=3, word_hidden_size=vocab_dim + addtional_feat)
        self.deep_attn_input_size = self.deep_attn.rnn_input_size
        self.deep_attn_output_size = self.deep_attn.output_size

        # Question understanding and compression
        self.high_lvl_ques_rnn , high_lvl_ques_rnn_output_size = RNN_from_opt(ques_rnn_output_size * opt['in_rnn_layers'], 
            opt['highlvl_hidden_size'], num_layers = opt['question_high_lvl_rnn_layers'], concat_rnn = True)

        self.after_deep_attn_size = self.deep_attn_output_size + self.deep_attn_input_size + addtional_feat + vocab_dim
        self.self_attn_input_size = self.after_deep_attn_size
        self_attn_output_size = self.deep_attn_output_size        

        # Self attention on context
        self.highlvl_self_att = Attention(self.self_attn_input_size, opt['deep_att_hidden_size_per_abstr'], correlation_func=3)
        print('Self deep-attention input is {}-dim'.format(self.self_attn_input_size))

        self.high_lvl_context_rnn, high_lvl_context_rnn_output_size = RNN_from_opt(self.deep_attn_output_size + self_attn_output_size, 
            opt['highlvl_hidden_size'], num_layers = 1, concat_rnn = False)
        context_final_size = high_lvl_context_rnn_output_size

        print('Do Question self attention')
        self.ques_self_attn = Attention(high_lvl_ques_rnn_output_size, opt['query_self_attn_hidden_size'], correlation_func=3)
        
        ques_final_size = high_lvl_ques_rnn_output_size
        print('Before answer span finding, hidden size are', context_final_size, ques_final_size)

        # Question merging
        self.ques_merger = LinearSelfAttn(ques_final_size)
        self.get_answer = GetFinalScores(context_final_size, ques_final_size)

    '''
    x: 1 x x_len (word_ids)
    x_single_mask: 1 x x_len
    x_char: 1 x x_len x char_len (char_ids)
    x_char_mask: 1 x x_len x char_len
    x_features: batch_size x x_len x feature_len (5, if answer_span_in_context_feature; 4 otherwise)
    x_pos: 1 x x_len (POS id)
    x_ent: 1 x x_len (entity id)
    x_bert: 1 x x_bert_token_len
    x_bert_mask: 1 x x_bert_token_len
    x_bert_offsets: 1 x x_len x 2
    q: batch x q_len  (word_ids)
    q_mask: batch x q_len
    q_char: batch x q_len x char_len (char ids)
    q_char_mask: batch x q_len x char_len
    q_bert: 1 x q_bert_token_len
    q_bert_mask: 1 x q_bert_token_len
    q_bert_offsets: 1 x q_len x 2
    context_len: number of words in context (only one per batch)
    return: 
      score_s: batch x context_len
      score_e: batch x context_len
      score_no: batch x 1
      score_yes: batch x 1
      score_noanswer: batch x 1
    '''
    def forward(self, x, x_single_mask, x_char, x_char_mask, x_features, x_pos, x_ent, x_bert, x_bert_mask, x_bert_offsets, q, q_mask, q_char, q_char_mask, q_bert, q_bert_mask, q_bert_offsets, context_len):
        batch_size = q.shape[0]
        x_mask = x_single_mask.expand(batch_size, -1)
        x_word_embed = self.vocab_embed(x).expand(batch_size, -1, -1) # batch x x_len x vocab_dim
        ques_word_embed = self.vocab_embed(q) # batch x q_len x vocab_dim

        x_input_list = [dropout(x_word_embed, p=self.opt['dropout_emb'], training=self.drop_emb)] # batch x x_len x vocab_dim
        ques_input_list = [dropout(ques_word_embed, p=self.opt['dropout_emb'], training=self.drop_emb)] # batch x q_len x vocab_dim

        # contextualized embedding
        x_cemb = ques_cemb = None        
        if 'BERT' in self.opt:
            x_cemb = ques_cemb = None
            
            if 'BERT_LINEAR_COMBINE' in self.opt:
                x_bert_output = self.Bert(x_bert, x_bert_mask, x_bert_offsets, x_single_mask)
                x_cemb_mid = self.linear_sum(x_bert_output, self.alphaBERT, self.gammaBERT)
                ques_bert_output = self.Bert(q_bert, q_bert_mask, q_bert_offsets, q_mask)
                ques_cemb_mid = self.linear_sum(ques_bert_output, self.alphaBERT, self.gammaBERT)
                x_cemb_mid = x_cemb_mid.expand(batch_size, -1, -1)
            else:    
                x_cemb_mid = self.Bert(x_bert, x_bert_mask, x_bert_offsets, x_single_mask)
                x_cemb_mid = x_cemb_mid.expand(batch_size, -1, -1)
                ques_cemb_mid = self.Bert(q_bert, q_bert_mask, q_bert_offsets, q_mask)

            x_input_list.append(x_cemb_mid)
            ques_input_list.append(ques_cemb_mid)

        if 'CHAR_CNN' in self.opt:
            x_char_final = self.character_cnn(x_char, x_char_mask)
            x_char_final = x_char_final.expand(batch_size, -1, -1)
            ques_char_final = self.character_cnn(q_char, q_char_mask)
            x_input_list.append(x_char_final)
            ques_input_list.append(ques_char_final)
        
        x_prealign = self.pre_align(x_word_embed, ques_word_embed, q_mask)
        x_input_list.append(x_prealign) # batch x x_len x (vocab_dim + cdim + vocab_dim)

        x_pos_emb = self.pos_embedding(x_pos).expand(batch_size, -1, -1) # batch x x_len x pos_dim
        x_ent_emb = self.ent_embedding(x_ent).expand(batch_size, -1, -1) # batch x x_len x ent_dim
        x_input_list.append(x_pos_emb)
        x_input_list.append(x_ent_emb)
        x_input_list.append(x_features)  # batch x x_len x (vocab_dim + cdim + vocab_dim + pos_dim + ent_dim + feature_dim)

        x_input = torch.cat(x_input_list, 2) # batch x x_len x (vocab_dim + cdim + vocab_dim + pos_dim + ent_dim + feature_dim)
        ques_input = torch.cat(ques_input_list, 2) # batch x q_len x (vocab_dim + cdim)

        # Multi-layer RNN
        _, x_rnn_layers = self.context_rnn(x_input, x_mask, return_list=True, x_additional=x_cemb) # layer x batch x x_len x context_rnn_output_size
        _, ques_rnn_layers = self.ques_rnn(ques_input, q_mask, return_list=True, x_additional=ques_cemb) # layer x batch x q_len x ques_rnn_output_size

        # rnn with question only 
        ques_highlvl = self.high_lvl_ques_rnn(torch.cat(ques_rnn_layers, 2), q_mask) # batch x q_len x high_lvl_ques_rnn_output_size
        ques_rnn_layers.append(ques_highlvl) # (layer + 1) layers

        # deep multilevel inter-attention
        if x_cemb is None:
            x_long = x_word_embed
            ques_long = ques_word_embed
        else:
            x_long = torch.cat([x_word_embed, x_cemb], 2)          # batch x x_len x (vocab_dim + cdim)
            ques_long = torch.cat([ques_word_embed, ques_cemb], 2) # batch x q_len x (vocab_dim + cdim)

        x_rnn_after_inter_attn, x_inter_attn = self.deep_attn([x_long], x_rnn_layers, [ques_long], ques_rnn_layers, x_mask, q_mask, return_bef_rnn=True)
        # x_rnn_after_inter_attn: batch x x_len x deep_attn_output_size
        # x_inter_attn: batch x x_len x deep_attn_input_size

        # deep self attention
        if x_cemb is None:
            x_self_attn_input = torch.cat([x_rnn_after_inter_attn, x_inter_attn, x_word_embed], 2)
        else:
            x_self_attn_input = torch.cat([x_rnn_after_inter_attn, x_inter_attn, x_cemb, x_word_embed], 2)
            # batch x x_len x (deep_attn_output_size + deep_attn_input_size + cdim + vocab_dim)
        
        x_self_attn_output = self.highlvl_self_att(x_self_attn_input, x_self_attn_input, x_mask, x3=x_rnn_after_inter_attn, drop_diagonal=True)
        # batch x x_len x deep_attn_output_size

        x_highlvl_output = self.high_lvl_context_rnn(torch.cat([x_rnn_after_inter_attn, x_self_attn_output], 2), x_mask)
        # bach x x_len x high_lvl_context_rnn.output_size
        x_final = x_highlvl_output

        # question self attention  
        ques_final = self.ques_self_attn(ques_highlvl, ques_highlvl, q_mask, x3=None, drop_diagonal=True) # batch x q_len x high_lvl_ques_rnn_output_size

        # merge questions  
        q_merge_weights = self.ques_merger(ques_final, q_mask) 
        ques_merged = weighted_avg(ques_final, q_merge_weights) # batch x ques_final_size

        # predict scores
        score_s, score_e, score_no, score_yes, score_noanswer = self.get_answer(x_final, ques_merged, x_mask)
        return score_s, score_e, score_no, score_yes, score_noanswer
    
    '''
     input: 
      x_char: batch x word_num x char_num
      x_char_mask: batch x word_num x char_num
     output: 
       x_char_cnn_final:  batch x word_num x char_cnn_hidden_size
    '''
    def character_cnn(self, x_char, x_char_mask):
        x_char_embed = self.char_embed(x_char) # batch x word_num x char_num x char_dim
        batch_size = x_char_embed.shape[0]
        word_num = x_char_embed.shape[1]
        char_num = x_char_embed.shape[2]
        char_dim = x_char_embed.shape[3]
        x_char_cnn = self.char_cnn(x_char_embed.contiguous().view(-1, char_num, char_dim), x_char_mask) # (batch x word_num) x char_num x char_cnn_hidden_size
        x_char_cnn_final = self.maxpooling(x_char_cnn, x_char_mask.contiguous().view(-1, char_num)).contiguous().view(batch_size, word_num, -1) # batch x word_num x char_cnn_hidden_size
        return x_char_cnn_final

    def linear_sum(self, output, alpha, gamma):
        alpha_softmax = F.softmax(alpha)
        for i in range(len(output)):
            t = output[i] * alpha_softmax[i] * gamma
            if i == 0:
                res = t
            else:
                res += t

        res = dropout(res, p=self.opt['dropout_emb'], training=self.drop_emb)
        return res
