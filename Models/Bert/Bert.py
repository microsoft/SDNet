# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from Models.Bert.modeling import BertModel

'''
    BERT    
'''
class Bert(nn.Module):
    def __init__(self, opt):
        super(Bert, self).__init__()
        print('Loading BERT model...')
        self.BERT_MAX_LEN = 512
        self.linear_combine = 'BERT_LINEAR_COMBINE' in opt
        
        if 'BERT_LARGE' in opt:
            print('Using BERT Large model')
            model_file = os.path.join(opt['datadir'], opt['BERT_large_model_file'])
            print('Loading BERT model from', model_file)
            self.bert_model = BertModel.from_pretrained(model_file)
            #self.bert_model = BertModel.from_pretrained('bert-large-uncased')        
            self.bert_dim = 1024
            self.bert_layer = 24
        else:
            print('Using BERT base model')
            model_file = os.path.join(opt['datadir'], opt['BERT_model_file'])
            print('Loading BERT model from', model_file)
            self.bert_model = BertModel.from_pretrained(model_file)
            #self.bert_model = BertModel.from_pretrained('bert-base-cased')        
            self.bert_dim = 768
            self.bert_layer = 12
        self.bert_model.cuda()
        self.bert_model.eval()

        print('Finished loading')

    '''
        Input:
              x_bert: batch * max_bert_sent_len (ids)
              x_bert_mask: batch * max_bert_sent_len (0/1)
              x_bert_offset: batch * max_real_word_num * 2
              x_mask: batch * max_real_word_num
            Output:
              embedding: batch * max_real_word_num * bert_dim
    '''
    def forward(self, x_bert, x_bert_mask, x_bert_offset, x_mask):
        if self.linear_combine:
            return self.combine_forward(x_bert, x_bert_mask, x_bert_offset, x_mask)

        last_layers = []
        bert_sent_len = x_bert.shape[1]
        p = 0
        while p < bert_sent_len:
            all_encoder_layers, _ = self.bert_model(x_bert[:, p:(p + self.BERT_MAX_LEN)], token_type_ids=None, attention_mask=x_bert_mask[:, p:(p + self.BERT_MAX_LEN)]) # bert_layer * batch * max_bert_sent_len * bert_dim
            last_layers.append(all_encoder_layers[-1]) # batch * up_to_512 * bert_dim
            p += self.BERT_MAX_LEN

        bert_embedding = torch.cat(last_layers, 1)
        
        batch_size = x_mask.shape[0]
        max_word_num = x_mask.shape[1]
        output = Variable(torch.zeros(batch_size, max_word_num, self.bert_dim))
        for i in range(batch_size):
            for j in range(max_word_num):
                if x_mask[i, j] == 0:
                    continue
                st = x_bert_offset[i, j, 0]    
                ed = x_bert_offset[i, j, 1]
                # we can also try using st only, ed only
                if st + 1 == ed: # including st==ed
                    output[i, j, :] = bert_embedding[i, st, :]
                else:    
                    subword_ebd_sum = torch.sum(bert_embedding[i, st:ed, :], dim = 0)
                    if st < ed:
                        output[i, j, :] = subword_ebd_sum / float(ed - st) # dim 0 is st:ed

        output = output.cuda()        
        return output

    def combine_forward(self, x_bert, x_bert_mask, x_bert_offset, x_mask):
        all_layers = []

        bert_sent_len = x_bert.shape[1]
        p = 0
        while p < bert_sent_len:
            all_encoder_layers, _ = self.bert_model(x_bert[:, p:(p + self.BERT_MAX_LEN)], token_type_ids=None, attention_mask=x_bert_mask[:, p:(p + self.BERT_MAX_LEN)]) # bert_layer * batch * max_bert_sent_len * bert_dim
            all_layers.append(torch.cat(all_encoder_layers, dim = 2))  # batch * up_to_512 * (bert_dim * layer)
            p += self.BERT_MAX_LEN

        bert_embedding = torch.cat(all_layers, dim = 1) # batch * up_to_512 * (bert_dim * layer)
        batch_size = x_mask.shape[0]
        max_word_num = x_mask.shape[1]
        tot_dim = bert_embedding.shape[2]
        output = Variable(torch.zeros(batch_size, max_word_num, tot_dim))
        for i in range(batch_size):
            for j in range(max_word_num):
                if x_mask[i, j] == 0:
                    continue
                st = x_bert_offset[i, j, 0]    
                ed = x_bert_offset[i, j, 1]
                # we can also try using st only, ed only
                if st + 1 == ed: # including st==ed
                    output[i, j, :] = bert_embedding[i, st, :]
                else:    
                    subword_ebd_sum = torch.sum(bert_embedding[i, st:ed, :], dim = 0)
                    if st < ed:
                        output[i, j, :] = subword_ebd_sum / float(ed - st) # dim 0 is st:ed

        outputs = []
        for i in range(self.bert_layer):
            now = output[:, :, (i * self.bert_dim) : ((i + 1) * self.bert_dim)]
            now = now.cuda()
            outputs.append(now)

        return outputs
