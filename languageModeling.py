import torch
import torch.nn as nn
import torchtext

import datasets
import random
import math
from tqdm import tqdm
import argparse
import re
from nltk.tokenize import NLTKWordTokenizer
import functools
import numpy as np
import os
import pickle

from newModels import baselineLM, durationLM, durationLMAdapt, durationLMLayer
from utilsLM import *
from FPmodel import simpleLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--fixation', action='store_true')
parser.add_argument('--small_lr', action='store_true')
parser.add_argument('--change_fix', default='no', type=str, choices=['no', 'random', 'random_word', 'full', 'freq'])
parser.add_argument('--hidden_dim', type=int)
parser.add_argument('--n_params', type=int)
parser.add_argument('--n_hiddens', type=int)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--rnn', type=str)
parser.add_argument('--model', type=str, default='FG_P', choices=['FG_P', 'FG_L']) # fixation-guided-parallel/layer
args = parser.parse_args()

assert args.change_fix == 'no' or args.fixation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = datasets.load_dataset('wikitext','wikitext-2-raw-v1')

tokenizer = NLTKWordTokenizer()
tokenize_func = lambda example, tokenizer: {'tokens': tokenizer.tokenize(example['text'])} 
tokenized_dataset = dataset.map(tokenize_func, fn_kwargs={'tokenizer': tokenizer})
# print(tokenized_dataset['train'][88]['text'])

vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset['train']['tokens'], min_freq=3) 
vocab.insert_token('<unk>', 0)           
vocab.insert_token('<eos>', 1)            
vocab.set_default_index(vocab['<unk>'])   
print(len(vocab))                         
# print(vocab.get_itos()[:10])
#80.9
if args.fixation:
    FPmodel, FPvocab = torch.load('./FPmodels/model-%s-%s-uniD.pt'%('simpleLSTM', "TRT"))
    if args.change_fix == 'random_word':
        path = './FPdatasets/random_word_mapping'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                fix_mapping = pickle.load(f)   
        else: 
            assert not args.test    
            fix_mapping = dict([(k, random.randint(0,11)) for k in FPvocab.id2word.keys()])
            with open(path, 'wb') as f:
                pickle.dump(fix_mapping, f)
        FPvocab.fix_mapping = fix_mapping
    for split in tokenized_dataset:
        tokenized_dataset[split] = add_fix_duration(args.change_fix, tokenized_dataset[split], FPvocab, FPmodel, device)

batch_size = 64 if not args.test else 1
train_data, train_fix = get_data(tokenized_dataset['train'], vocab, batch_size, args.fixation)
valid_data, valid_fix = get_data(tokenized_dataset['validation'], vocab, batch_size, args.fixation)
test_data, test_fix = get_data(tokenized_dataset['test'], vocab, batch_size, args.fixation)

embedding_dim = 100             # 400 in the paper
               # 1150 in the paper
num_layers = None                   # 3 in the paper
dropout_rate_emb = 0.5    
dropout_rate_layer = 0.25          
tie_weights = True                  
lr = 1e-3 if not args.small_lr else 1e-4
print(f'learning rate {lr}')

if args.fixation:
    hidden_dim = args.hidden_dim     # h_f = h_n / sqrt(n_hiddens)
    if args.model == 'FG_P':
        assert args.rnn in ['v1', 'lstmv1',]
        n_hiddens = args.n_hiddens
        hidden_dim = determine_hidden_dim(args.n_params, durationLMAdapt, [args.rnn, vocab, embedding_dim, hidden_dim, n_hiddens, dropout_rate_emb, dropout_rate_layer, tie_weights, None, None, None, None, False, args.n_layers, False])
        model = durationLMAdapt(args.rnn, vocab, embedding_dim, hidden_dim, n_hiddens, dropout_rate_emb, dropout_rate_layer, tie_weights, None, None, None, None, adapt=False, num_layers=args.n_layers, pretrained_embedding=(not args.test)).to(device)
        model_name = model._get_name() + '-%s-%dx%d'%(args.rnn, n_hiddens, hidden_dim)
    elif args.model == 'FG_L':
        assert args.rnn in ['rnn', 'lstm']                              
        hidden_dim = determine_hidden_dim(args.n_params, durationLMLayer, [args.rnn, vocab, embedding_dim, hidden_dim, dropout_rate_emb, dropout_rate_layer, tie_weights, None, None, None, None,  False, args.n_layers, False])
        model = durationLMLayer(args.rnn, vocab, embedding_dim, hidden_dim, dropout_rate_emb, dropout_rate_layer, tie_weights, None, None, None, None,  adapt=False, num_layers=args.n_layers, pretrained_embedding=(not args.test)).to(device)
        model_name = model._get_name() + '-%s-%dlayer-%d'%(args.rnn, args.n_layers, hidden_dim)
else:
    hidden_dim = args.hidden_dim
    assert args.rnn  in ['rnn', 'lstm']
    hidden_dim = determine_hidden_dim(args.n_params, baselineLM, [args.rnn, vocab, embedding_dim, hidden_dim, args.n_layers, dropout_rate_emb, dropout_rate_layer, tie_weights, False])
    model = baselineLM(args.rnn, vocab, embedding_dim, hidden_dim, args.n_layers, dropout_rate_emb, dropout_rate_layer, tie_weights, pretrained_embedding=(not args.test)).to(device)
    model_name = model._get_name() + '-%s-%dlayer-%d'%(args.rnn, args.n_layers, hidden_dim)

print(model)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
criterion = nn.CrossEntropyLoss()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Using hidden dim = {hidden_dim}')
print(f'The model has {num_params:,} trainable parameters, {num_params - model.embedding.weight.numel():,} excluding embedding')
if args.change_fix != 'no':
    print(f'===== using {args.change_fix} fixation values! ====')

def train(model, data, optimizer, criterion, batch_size, seq_len_param, clip, device, fix_duration=None):
    
    epoch_loss = 0
    model.train()

    base_seq_len, std_seq_len = seq_len_param

    hidden = model.init_hidden(batch_size, device)
    total_len = data.size(1)
    idx = 0
    pbar = tqdm(total=total_len-1, desc='Training: ',leave=False)
    while idx < total_len - 1:
    # for idx in tqdm(range(0, total_len - 1, seq_len), desc='Training: ',leave=False):

        seq_len = max(base_seq_len//4, round(np.random.normal(base_seq_len if np.random.uniform() < 0.95 else base_seq_len//2, std_seq_len)))

        hidden = model.detach_hidden(hidden)

        assert fix_duration == None or data.size() == fix_duration.size()
        src, target, fix = get_batch(data, seq_len, total_len, idx, fix_duration)
        src, target, fix = src.to(device), target.to(device), fix.to(device)
       
        prediction, hidden =  model(src, hidden, fix) if fix_duration != None else model(src, hidden)  

        prediction = prediction.view(-1, prediction.size(2))   
        loss = criterion(prediction, target.contiguous().view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * target.size(1)

        idx += target.size(1)
        pbar.update(target.size(1))
    return epoch_loss / (total_len-1)

def evaluate(model, data, criterion, batch_size, seq_len_param, device, fix_duration=None):

    epoch_loss = 0
    model.eval()
    
    base_seq_len, std_seq_len = seq_len_param

    hidden = model.init_hidden(batch_size, device)
    total_len = data.size(1)
    with torch.no_grad():
        for idx in range(0, total_len - 1, base_seq_len):
            hidden = model.detach_hidden(hidden)

            assert fix_duration == None or data.size() == fix_duration.size()
            src, target, fix = get_batch(data, base_seq_len, total_len, idx, fix_duration)
            src, target, fix = src.to(device), target.to(device), fix.to(device)

            prediction, hidden =  model(src, hidden, fix) if fix_duration != None else model(src, hidden) 

            prediction = prediction.view(-1, prediction.size(2))   
            loss = criterion(prediction, target.contiguous().view(-1))

            epoch_loss += loss.item() * target.size(1)
    return epoch_loss / (total_len-1)

n_epochs = 50
base_seq_len, std_seq_len = 100, 5
seq_len_param = (base_seq_len, std_seq_len)
# seq_len = round(np.random.normal(base_seq_len if np.random.uniform() < 0.95 else base_seq_len//2, std_seq_len))
clip = 0.25

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

if args.test:
    model.load_state_dict(torch.load(f'./LMmodels/best-val-{model_name}_lm.pt',  map_location=device))
    test_loss = evaluate(model, test_data, criterion, batch_size, seq_len_param, device, test_fix if args.fixation else None)
    print(f'Test Perplexity: {math.exp(test_loss):.3f}')
else:
    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        train_loss = train(model, train_data, optimizer, criterion, 
                    batch_size, seq_len_param, clip, device, train_fix if args.fixation else None)
        valid_loss = evaluate(model, valid_data, criterion, batch_size, 
                    seq_len_param, device, valid_fix if args.fixation else None)
        
        lr_scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'./LMmodels/best-val-{model_name}_lm.pt')

        print(f'Epoch: {epoch} \tTrain Perplexity: {math.exp(train_loss):.3f} \tValid Perplexity: {math.exp(valid_loss):.3f}')
