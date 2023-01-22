import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm import tqdm
import random
import re

def add_fix_duration(change_fix, dataset, vocab, FPmodel, device):
    data = []                                        
    for example in dataset:
        if example['tokens']:           
            token_ids = [vocab.get_id( re.sub('[=@]', '-', re.sub(r'\d','0',token.lower())) ) for token in example['tokens']]
            data.append(token_ids)
    
    # predict in batch
    pred_fix = []
    if change_fix == 'random':
        for token_ids in data:
            pred_fix.append( [random.randint(0, 11) for i in token_ids] + [random.randint(0, 11)] ) # add one for eos
    elif change_fix == 'random_word':
        for token_ids in data:
            pred_fix.append( [vocab.fix_mapping[i] for i in token_ids] + [1,] )
    elif change_fix == 'full':
        for token_ids in data:
            pred_fix.append( [11,] * (len(token_ids)+1) )
    elif change_fix == 'freq':
        # more frequent, less fixation. first pad, second unknown
        total_count = sum( [item[1] for item in vocab.ordered_lis[2:]])
        mapping = {}
        fix = 0
        count = 0
        interval = total_count // 12
        for word, freq in vocab.ordered_lis[2:]:
            mapping[word] = fix
            count += freq
            if count >= interval:
                fix  = min(fix + 1, 11)
                count -= interval
        mapping['[UNK]'] = 11
        for token_ids in data:
            pred_fix.append( [mapping[vocab.get_word(i)] for i in token_ids] + [1,] )
    else: 
        FPmodel = FPmodel.to(device)
        FPmodel.eval()
        print('predicting fixation duration')
        batch_size = 128
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]

            max_word_num = max(list(map(lambda x:len(x),batch)))
            input_ids=list(map(lambda x: x+[vocab.pad_id]*(max_word_num-len(x)), batch))
            input_ids=torch.LongTensor(input_ids).to(device)

            with torch.no_grad():
                pred_fix.extend( torch.round(FPmodel(input_ids)[0]).long().tolist() )
        del FPmodel
    assert len(data) == len(pred_fix)

    new_field = []
    idx = 0
    for example in dataset:
        if example['tokens']:
            if change_fix == 'no':
                new_field.append(pred_fix[idx][:len(example['tokens'])] + [3,])
            else:
                assert len(pred_fix[idx]) == len(example['tokens']) + 1
                new_field.append(pred_fix[idx])
            idx += 1
        else:
            new_field.append([])
    dataset = dataset.add_column('fix_duration', new_field)
    return dataset


def expand_vocab_FP(FPmodel, FPvocab):
    new_tokens = ['[EOS]', '=', '@']
    for token in new_tokens:
        if FPvocab.get_id(token) == FPvocab.unk_id:
            idx = FPvocab.get_len()
            FPvocab.id2word[idx] = token
            FPvocab.word2id[token] = idx
            print(token, '\tadded to FPvocab')

    original_len, emb_dim = FPmodel.word_emb.weight.data.size()
    expanded_word_emb = nn.Embedding(FPvocab.get_len(), emb_dim, padding_idx=FPvocab.pad_id)
    expanded_word_emb.weight.data[:original_len] = FPmodel.word_emb.weight.data.detach()
    FPmodel.word_emb = expanded_word_emb

    return FPmodel, FPvocab

def add_fix_src(dataset, FPvocab):
    fix_src = []
    for example in dataset:
        if example['tokens']:        
            token_ids = [FPvocab.get_id( re.sub(r'\d','0',token.lower()) ) for token in example['tokens']] + [FPvocab.get_id('[EOS]')]
            fix_src.append(token_ids)
        else:
            fix_src.append([])
    dataset = dataset.add_column('fix_duration', fix_src)
    return dataset



def get_data(dataset, vocab, batch_size, fixation):
    data = []
    fix_duration = []                                 
    for example in dataset:
        if example['tokens']:           
            tokens = [vocab[token] for token in example['tokens']] + [vocab['<eos>']]
            data.extend(tokens)      
            if fixation:
                fix_duration.extend( example['fix_duration'] )     
    assert not fixation or len(data) == len(fix_duration)  

    data = torch.LongTensor(data)
    n_cols = data.size(0) // batch_size 
    data = data[:n_cols * batch_size]                       
    data = data.view(batch_size, n_cols)    
    if fixation:
        fix_duration = torch.LongTensor(fix_duration)     
        fix_duration = fix_duration[:n_cols * batch_size]                       
        fix_duration = fix_duration.view(batch_size, n_cols)
    return data, fix_duration

def get_data_FP(dataset, vocab, batch_size, with_var=False):
    data = []  
    labels = []                            
    for i in range(len(dataset)):
        if dataset.input_tokens[i]:           
            tokens = [vocab[token] for token in dataset.input_tokens[i]] + [vocab['<eos>']]
            l = dataset.labels[i] +  [(0, float('inf'))] if with_var else [-1,]
            data.extend(tokens)
            labels.extend(l)        

    assert len(data) == len(labels)
    data = torch.LongTensor(data)
    labels = torch.tensor(labels) if with_var else torch.LongTensor(labels)

    n_cols = data.size(0) // batch_size 
    data = data[:n_cols * batch_size]    
    data = data.view(batch_size, n_cols)  

    labels = labels[:n_cols * batch_size]
    labels = labels.view(batch_size, n_cols, -1) if with_var else labels.view(batch_size, n_cols)    
    return data, labels

def get_batch(data, seq_len, total_len, idx, fix_duration):
    end = min(idx+seq_len, total_len-1)
    src = data[:, idx:end]                   
    target = data[:, idx+1:end+1]   
    fix = torch.zeros(1)
    if fix_duration != None:
        fix = fix_duration[:, idx:end]          # can try idx+1:end+1
    return src, target, fix

def get_batch_FP(data, label, seq_len, idx):
    total_len = data.size(1)
    end = min(idx+seq_len, total_len)
    src = data[:, idx:end]                   
    target = label[:, idx:end]
    if end == total_len:
        end = 0
    return src, target, end

def determine_hidden_dim(n_params, model_class, params):
    suggested_dim = params[3]
    dim_space = sorted(list(range(10,5000,10)) + list(map(lambda x: 2**x, range(1,13))))
    idx = dim_space.index(suggested_dim)
    best_dim = None
    smallest_dif = float('inf')
    best_dim_param = 0
    for i in range(idx-10, idx+10):
        params[3] = dim_space[i]
        model = model_class(*params)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_params -= model.embedding.weight.numel()
        # if hasattr(model, 'FPmodel'):
        #     num_params -= model.FPmodel.word_emb.weight.numel()
        # dif = min(list(map(lambda x: abs(num_params-x), [1e6, 4e6, 16e6])))
        dif = abs(num_params-n_params*1e6)
        if dif < smallest_dif:
            best_dim = dim_space[i]
            best_dim_param = num_params
            smallest_dif = dif
        del model
    print(f'Using hidden dim = {best_dim} num_params = {best_dim_param}')
    return best_dim


def examine_fix(model, data, batch_size, device, fix_duration, FPvocab, use_same_emb):

    model.eval()
    if hasattr(model, 'norm_layer'):
        print(list((model.norm_layer.named_parameters())))
    if hasattr(model, 'norm_bias'):
        print(model.norm_bias)
    if hasattr(model.FPmodel, 'a'):
        print(model.FPmodel.a)
    if hasattr(model.FPmodel, 'b'):
        print(model.FPmodel.b)
    base_seq_len = 100

    # hidden, fix_hidden = model.init_hidden(batch_size, device)
    # # total_len = data.size(1)
    # total_len = 500
    # with torch.no_grad():
    #     for idx in range(0, total_len - 1, base_seq_len):

    #         src, target, fix = get_batch(data, base_seq_len, total_len, idx, fix_duration)
    #         src, target, fix = src.to(device), target.to(device), fix.to(device)
    #         if use_same_emb:
    #             fix = src
    #             func = FPvocab.lookup_token
    #         else:
    #             func = FPvocab.get_word
    #         fix_pred, fix_hidden = model.fix_predict(fix, fix_hidden)
    #         fix_pred = torch.round(fix_pred).long()
            
    #         i = random.randint(0, len(fix)-1)
    #         tokens = list(map(func ,fix[i].tolist()))
    #         print(list(zip(tokens, fix_pred[i].tolist())))
        
    return 
