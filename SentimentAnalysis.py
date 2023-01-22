from SAdatasets import SentimentAnalysisDataset, collate_batch
from SAmodel import baselineSA, durationLMAdapt
from utilsSA import add_fix_duration
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from FPmodel import read_embedding
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fixation', action='store_true')
parser.add_argument('--adapt', action='store_true')
parser.add_argument('--multitask', action='store_true')
parser.add_argument('--rnn', type=str, choices=['rnn', 'lstm'], default='lstm')
parser.add_argument('--FP_fix', action='store_true')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using: ', device)


BATCH_SIZE=32 
LR=0.001
CLIP=5.
use_fixation = args.fixation
rnn = args.rnn  #'lstm' # or 'rnn'
num_runs = 3 # run n times, then average the result
# 100 3 0.5645  

if use_fixation:
    adapt = args.adapt
    n_hiddens = [3, 6, 12]

assert not args.adapt or (use_fixation and args.adapt)
assert not args.multitask or (args.adapt and args.multitask)
assert not args.multitask or (not args.FP_fix and args.multitask)


hidden_dim = [30, 50, 100, 200]
n_layers = [1,]
hyper_params = [(h, l) for h in hidden_dim for l in n_layers]

if use_fixation:
    hyper_params = [p + (n,) for p in hyper_params for n in n_hiddens]
else:
    hyper_params = [p + (None,) for p in hyper_params]

#  using true fixation duration
#  6 50  0.7980

# using adaptive
# 6 50 0.8333
# 3 50 0.8485
# 12 30 0.8333

# baseline
# 30 0.8367
# 50 0.8485
# 100 0.8502 * 
# 200 0.8434

SAdataset = SentimentAnalysisDataset(test_proportion=0.2, redo=False)
if args.FP_fix:
    FPmodel, FPvocab = torch.load('./FPmodels/model-%s-%s-uniD.pt'%('simpleLSTM', "TRT"))
    add_fix_duration(SAdataset.train_set, FPvocab, FPmodel, device)
    add_fix_duration(SAdataset.test_set, FPvocab, FPmodel, device)

train_loader = DataLoader(SAdataset.train_set, BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(SAdataset.test_set, BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

"""load embedding"""
word2emb = read_embedding()

all_result = []
for hidden_dim, n_layers, n_hiddens in hyper_params:
    print('testing', hidden_dim, n_layers, n_hiddens )
    result = []
    for run in range(num_runs):
        if not use_fixation:
            model = baselineSA(SAdataset.mapping, lstm_dim=hidden_dim, bidirectional=False,pretrained_embedding=word2emb, rnn=rnn, num_layers=n_layers).to(device=device)
        else:
            model =  durationLMAdapt('lstmv1' if rnn == 'lstm' else 'v1', SAdataset.mapping, 100, hidden_dim, n_hiddens, 0.5, 0.25, \
                        None, None, 4.0, True, True, adapt=adapt, num_layers=n_layers, pretrained_embedding=word2emb).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), LR)
        loss_func = nn.CrossEntropyLoss()
        if args.multitask:
            FP_criterion = nn.MSELoss()
        print(model)
        print(model._get_name())
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_params = num_params - model.embedding.weight.numel()
        print(f'The model has {num_params:,} trainable parameters, {model_params:,} excluding embedding')
        if model_params > 2000000:
            print('model is too large, next one')
            continue

        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

        best_acc = 0
        for epoch in range(30):
            model.train()
            train_loss = []
            for i, (input_ids, fix_duration, input_len, labels) in enumerate(train_loader):
                input_ids = input_ids.to(device=device)
                labels = labels.to(device=device)
                fix_duration = fix_duration.to(device=device)
                input_len = input_len.to(device=device)

                pred = model(input_ids, input_len, fix_duration)
                loss = loss_func(pred, labels)

                if args.multitask:

                    FP_prediction, _ = model.FPmodel(input_ids)
                    FP_prediction = FP_prediction[input_ids != 0]
                    fix_duration = fix_duration[input_ids != 0]
                    FP_loss = FP_criterion(FP_prediction, fix_duration.float())
                    # print(loss, 0.003*FP_loss)
                    loss = loss + 0.001 * FP_loss

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(),CLIP)
                optimizer.step()

                train_loss.append(loss.item())

            train_loss = sum(train_loss) / len(train_loss)

            with torch.no_grad():
                model.eval()
                test_correct = 0
                test_num = 0
                for i, (input_ids, fix_duration, input_len, labels) in enumerate(test_loader):
                    input_ids = input_ids.to(device=device)
                    labels = labels.to(device=device)
                    fix_duration = fix_duration.to(device=device)
                    input_len = input_len.to(device=device)

                    pred = model(input_ids, input_len, fix_duration)
                    pred = pred.argmax(dim=1)

                    test_correct += (pred == labels).sum().item()
                    test_num += pred.numel()



                test_acc = test_correct / test_num

            print('epoch %d:  train loss: %.4f  test acc: %.4f'%(epoch, train_loss, test_acc)) #optimizer.param_groups[0]['lr']
            # lr_scheduler.step(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
        print('best acc: ', best_acc)
        result.append(best_acc)

    print('==============================')
    print('average best acc: ', sum(result) / num_runs)
    all_result.append((hidden_dim, n_layers, n_hiddens, sum(result) / num_runs))
    print(all_result)

all_result.sort(key=lambda x: -x[3])
for item in all_result:
    print(str(item[:3]) + '\t %.5f'%item[3] )
