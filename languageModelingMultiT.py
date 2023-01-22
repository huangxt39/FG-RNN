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

from newModels import baselineLM, durationLM, durationLMAdapt, durationLMLayer
from utilsLM import *
from FPmodel import simpleLSTM
from FPdataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--small_lr', action='store_true')
parser.add_argument('--hidden_dim', type=int)
parser.add_argument('--n_params', type=int)
parser.add_argument('--n_hiddens', type=int)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--rnn', type=str)
parser.add_argument('--feature', type=str)
parser.add_argument('--alpha', type=float)
parser.add_argument('--rescale_rate', type=float, default=4.0)
parser.add_argument('--use_norm', action='store_true')
parser.add_argument('--multitask', action='store_true')
parser.add_argument('--model', type=str, default='FG_P', choices=['FG_P', 'FG_L']) # fixation-guided-parallel/layer
args = parser.parse_args()


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
batch_size = 64 if not args.test else 1

if args.multitask:
    # FP_dataset = fixationPredictionDataset('TRT', average=True, test_proportion=0.0, shuffle=False)
    FP_dataset = fixationPredictionDatasetVar(args.feature, test_proportion=0.0, shuffle=False)
    print(f'using feature: {args.feature}')
    FP_data, FP_label = get_data_FP(FP_dataset.train_set, vocab, batch_size, with_var=True)
    print(FP_label[:,:,1].isnan().sum())
    FP_label[:,:,1] /= FP_label[:,:,0].var()
    FP_label[:,:,0] = (FP_label[:,:,0] - FP_label[:,:,0].mean()) / FP_label[:,:,0].std()
    print('FP_label size', FP_label.size())
else:
    FP_data = None
    FP_label = None

train_data, _ = get_data(tokenized_dataset['train'], vocab, batch_size, False)
valid_data, _ = get_data(tokenized_dataset['validation'], vocab, batch_size, False)
test_data, _ = get_data(tokenized_dataset['test'], vocab, batch_size, False)


embedding_dim = 100             # 400 in the paper
               # 1150 in the paper
dropout_rate_emb = 0.5    
dropout_rate_layer = 0.25          
tie_weights = True                  
lr = 1e-3 if not args.small_lr else 1e-4   

n_hiddens = args.n_hiddens
hidden_dim = args.hidden_dim     # h_f = h_n / sqrt(n_hiddens)

use_same_emb = True
use_norm = args.use_norm

FPmodel = None
if args.model == 'FG_P':
    hidden_dim = determine_hidden_dim(args.n_params, durationLMAdapt, [args.rnn, vocab, embedding_dim, hidden_dim, n_hiddens, dropout_rate_emb, dropout_rate_layer, tie_weights, FPmodel, args.rescale_rate, use_same_emb, use_norm, True, args.n_layers, False])
    model = durationLMAdapt(args.rnn, vocab, embedding_dim, hidden_dim, n_hiddens, dropout_rate_emb, dropout_rate_layer, tie_weights, FPmodel, args.rescale_rate, use_same_emb, use_norm, True, num_layers=args.n_layers, pretrained_embedding=(not args.test)).to(device)
    model_name = model._get_name() + '-%s-%dx%d-FG_P'%(args.rnn, n_hiddens, hidden_dim)  
elif args.model == 'FG_L':
    hidden_dim = determine_hidden_dim(args.n_params, durationLMLayer, [args.rnn, vocab, embedding_dim, hidden_dim, dropout_rate_emb, dropout_rate_layer, tie_weights, FPmodel, args.rescale_rate, use_same_emb, use_norm, True, args.n_layers, False])
    model = durationLMLayer(args.rnn, vocab, embedding_dim, hidden_dim, dropout_rate_emb, dropout_rate_layer, tie_weights, FPmodel, args.rescale_rate, use_same_emb, use_norm, True, num_layers=args.n_layers, pretrained_embedding=(not args.test)).to(device)
    model_name = model._get_name() + '-%s-%dlayer-%d-FG_L'%(args.rnn, args.n_layers, hidden_dim)  

print(model)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
criterion = nn.CrossEntropyLoss()
FP_criterion = nn.MSELoss(reduction='none')
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Using hidden dim = {hidden_dim}')
print(f'The model has {num_params:,} trainable parameters, {num_params - model.embedding.weight.numel():,} excluding embedding')

if args.multitask:
    print(f'===== fixation prediction are training simultaneously ====')

def train(model, data, FP_data, FP_target, optimizer, criterion, FP_criterion, ignore_idx, batch_size, seq_len_param, clip, alpha, device, multitask=True):
    
    epoch_loss = 0
    FP_epoch_loss = 0
    FP_num = 0
    model.train()

    base_seq_len, std_seq_len = seq_len_param

    hidden = model.init_hidden(batch_size, device)
    if multitask:
        FP_hidden = model.FPmodel.init_hidden(batch_size, device)
    total_len = data.size(1)
    idx = 0
    FP_idx = 0
    pbar = tqdm(total=total_len-1, desc='Training: ',leave=False)
    while idx < total_len - 1:
    # for idx in tqdm(range(0, total_len - 1, seq_len), desc='Training: ',leave=False):

        seq_len = max(base_seq_len//4, round(np.random.normal(base_seq_len if np.random.uniform() < 0.95 else base_seq_len//2, std_seq_len)))

        hidden = model.detach_hidden(hidden)
        if multitask:
            FP_hidden = model.FPmodel.detach_hidden(FP_hidden)

        if multitask:
            FP_src, FP_target, FP_idx = get_batch_FP(FP_data, FP_label, seq_len, FP_idx)
            FP_src, FP_target = FP_src.to(device), FP_target.to(device)

            FP_prediction, FP_hidden = model.FPmodel(FP_src, FP_hidden)
            if FP_label.dim() == 2:
                FP_prediction = FP_prediction[FP_src != ignore_idx]
                FP_target = FP_target[FP_src != ignore_idx]
                FP_loss = FP_criterion(FP_prediction, FP_target.float())
            elif FP_label.dim() == 3:
                FP_loss = (FP_criterion(FP_prediction, FP_target[:,:,0]) / (FP_target[:,:,1] + 0.1)).mean()

        # FP_loss.backward()
        # print(FP_loss)
        # print(torch.linalg.norm(model.FPmodel.lstm.weight_hh_l0.grad))
        # optimizer.zero_grad()
        

        src, target, _ = get_batch(data, seq_len, total_len, idx, None)
        src, target = src.to(device), target.to(device)
       
        prediction, hidden = model(src, hidden, None)  
        prediction = prediction.view(-1, prediction.size(2))   
        loss = criterion(prediction, target.contiguous().view(-1))
        
        # loss.backward()
        # print(loss)
        # print(torch.linalg.norm(model.FPmodel.lstm.weight_hh_l0.grad))
        # optimizer.zero_grad()

        
        
        # total_loss = alpha * loss + (1 - alpha) * FP_loss
        if multitask:
            total_loss = loss + alpha * FP_loss
        else:
            total_loss = loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if multitask:
            FP_epoch_loss += FP_loss.item() * FP_prediction.numel()
            FP_num += FP_prediction.numel()
        else:
            FP_num = 1
        epoch_loss += loss.item() * target.size(1)

        idx += target.size(1)
        pbar.update(target.size(1))
    return epoch_loss / (total_len-1), FP_epoch_loss / FP_num

def evaluate(model, data, criterion, batch_size, seq_len_param, device):

    epoch_loss = 0
    model.eval()
    
    base_seq_len, std_seq_len = seq_len_param

    hidden = model.init_hidden(batch_size, device)
    total_len = data.size(1)
    with torch.no_grad():
        for idx in range(0, total_len - 1, base_seq_len):
            hidden = model.detach_hidden(hidden)

            src, target, _ = get_batch(data, base_seq_len, total_len, idx, None)
            src, target = src.to(device), target.to(device)

            prediction, hidden =  model(src, hidden, None)

            prediction = prediction.view(-1, prediction.size(2))   
            loss = criterion(prediction, target.contiguous().view(-1))

            epoch_loss += loss.item() * target.size(1)
    return epoch_loss / (total_len-1)

def FP_train(model, FP_data, FP_target, FP_criterion, ignore_idx, batch_size, seq_len_param, clip, device):
    
    FP_epoch_loss = 0
    FP_num = 0
    model.FPmodel.train()

    base_seq_len, std_seq_len = seq_len_param

    optimizer = torch.optim.Adam(model.FPmodel.parameters(), lr=1e-3)
    FP_hidden = model.FPmodel.init_hidden(batch_size, device)
    total_len = FP_data.size(1)
    FP_idx = 0
    pbar = tqdm(total=total_len, desc='Training: ',leave=False)
    while FP_idx < total_len:

        seq_len = max(base_seq_len//4, round(np.random.normal(base_seq_len if np.random.uniform() < 0.95 else base_seq_len//2, std_seq_len)))

        FP_hidden = model.FPmodel.detach_hidden(FP_hidden)

        end = min(FP_idx+seq_len, total_len)
        FP_src = FP_data[:, FP_idx:end].to(device)             
        FP_target = FP_label[:, FP_idx:end].to(device)

        FP_prediction, FP_hidden = model.FPmodel(FP_src, FP_hidden)
        if FP_label.dim() == 2:
            FP_prediction = FP_prediction[FP_src != ignore_idx]
            FP_target = FP_target[FP_src != ignore_idx]
            FP_loss = FP_criterion(FP_prediction, FP_target.float())
        elif FP_label.dim() == 3:
            # print('---')
            # print((FP_target[:,:,1] == 0).sum())
            # print(((FP_criterion(FP_prediction, FP_target[:,:,0]) / FP_target[:,:,1])==torch.inf).sum())
            # print((FP_criterion(FP_prediction, FP_target[:,:,0]) / FP_target[:,:,1]).size())
            # print(FP_target[:,:,1])
            FP_loss = (FP_criterion(FP_prediction, FP_target[:,:,0]) / (FP_target[:,:,1] + 0.1)).mean()

        optimizer.zero_grad()
        FP_loss.backward()
        # print(FP_loss)
        # print(torch.linalg.norm(model.FPmodel.lstm.weight_hh_l0.grad))
        torch.nn.utils.clip_grad_norm_(model.FPmodel.parameters(), clip)
        optimizer.step()

        FP_epoch_loss += FP_loss.item() * FP_prediction.numel()
        FP_num += FP_prediction.numel()

        pbar.update(end - FP_idx)
        FP_idx += end
    return FP_epoch_loss / FP_num

n_epochs = 50
base_seq_len, std_seq_len = 100, 5
seq_len_param = (base_seq_len, std_seq_len)
# seq_len = round(np.random.normal(base_seq_len if np.random.uniform() < 0.95 else base_seq_len//2, std_seq_len))
clip = 0.25

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

if args.test:
    model.load_state_dict(torch.load(f'./LMmodels/best-val-{model_name}_lm.pt',  map_location=device))
    examine_fix(model, test_data, batch_size, device, None, vocab, use_same_emb)
    test_loss = evaluate(model, test_data, criterion, batch_size, seq_len_param, device)
    print(f'Test Perplexity: {math.exp(test_loss):.3f}')
else:
    best_valid_loss = float('inf')

    # for epoch in range(100):
    #     FP_train_loss = FP_train(model, FP_data, FP_label, FP_criterion, vocab['<eos>'], batch_size, seq_len_param, clip, device)
    #     print(f'Epoch: {epoch} \tTrain FP loss: {FP_train_loss:.2f}')

    for epoch in range(n_epochs):
        # alpha = 1 - 1 / (epoch + 2)
        # alpha = 1 - max(0.9 - epoch*0.1, 0)
        train_loss, FP_train_loss = train(model, train_data, FP_data, FP_label, optimizer, criterion, FP_criterion, vocab['<eos>'],
                    batch_size, seq_len_param, clip, args.alpha, device, args.multitask)
        examine_fix(model, test_data, batch_size, device, None, vocab, use_same_emb)
        valid_loss = evaluate(model, valid_data, criterion, batch_size, 
                    seq_len_param, device)
        
        lr_scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'./LMmodels/best-val-{model_name}_lm.pt')

        print(f'Epoch: {epoch} \tTrain Perplexity: {math.exp(train_loss):.3f} \tTrain FP loss: {FP_train_loss:.2f} \tValid Perplexity: {math.exp(valid_loss):.3f}')
