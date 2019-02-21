# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import re
from Utils.Constants import *
import spacy
import torch
import torch.nn.functional as F
import unicodedata
import sys
from torch.autograd import Variable
nlp = spacy.load('en', parser = False)

# normalize sentence
def normalize_text(text):
    return unicodedata.normalize('NFD', text)
 
def space_extend( matchobj):
    return ' ' + matchobj.group(0) + ' '

# get rid of punctuation stuff and stripping
def pre_proc(text):
    text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t', space_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text 

# get a set of vocabulary
def load_glove_vocab(file, wv_dim, to_lower = True):
    glove_vocab = set()
    print('Loading glove vocabulary from ' + file)
    lineCnt = 0
    with open(file, encoding = 'utf-8') as f:
        for line in f:
            # delete!!!
            #if lineCnt == 20000:
            #    print('delete!')
            #    break

            lineCnt = lineCnt + 1
            if lineCnt % 100000 == 0:
                print('.', end = '',flush=True)
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if to_lower:
                token = token.lower()
            glove_vocab.add(token) 

    print('\n')
    print('%d words loaded from Glove\n' % len(glove_vocab))
    return glove_vocab

def token2id(docs, vocab, unk_id=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids

def char2id(docs, char_vocab, unk_id=None):
    c2id = {c: i for i, c in enumerate(char_vocab)}
    ids = [[[c2id["<STA>"]] + [c2id[c] if c in c2id else unk_id for c in w] + [c2id["<END>"]] for w in doc] for doc in docs]
    return ids

def removeInvalidChar(sentence):
    ordId = list(sentence.encode('utf-8', errors='ignore'))
    ordId = [x for x in ordId if x >= 0 and x < 256]
    return ''.join([chr(x) for x in ordId])

def makeVariable(x, use_cuda):
    if use_cuda:
        x = x.pin_memory()
        return Variable(x.cuda(async = True), requires_grad = False)
    else:
        return Variable(x, requires_grad = False)

'''
Input:
 nlp is an instance of spacy
 sentence is a string

Output:
 A list of tokens, entity and POS tags
'''
def spacyTokenize(sentence, vocab_ent=None, vocab_tag=None):
    sentence = sentence.lower()
    sentence = pre_proc(sentence)
    raw_tokens = nlp(sentence)
    tokens = [normalize_text(token.text) for token in raw_tokens if not token.is_punct | token.is_space]
    ent = None
    if vocab_ent is not None:
        ent = [token2id(token.ent_type_, vocab_ent) + 1 for token in raw_tokens if not token.is_punct | token.is_space]

    tag = None
    if vocab_tag is not None:
        tag = [token2id(token.tag_, vocab_tag) + 1 for token in raw_tokens if not token.is_punct | token.is_space]    

    return tokens, ent, tag
