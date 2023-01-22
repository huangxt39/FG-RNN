from FPmodel import simpleLSTM, feedBackLSTM
from FPdataset import fixationPredictionDataset, collate_batch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using: ', device)

FPdataset = fixationPredictionDataset('TRT', average=True)

BATCH_SIZE=32 
LR=0.001
CLIP=5.
lstm_dim=150
n_layers=1

print(BATCH_SIZE, LR, lstm_dim, n_layers)



train_loader = DataLoader(FPdataset.train_set, BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(FPdataset.test_set, BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

model = simpleLSTM(FPdataset.mapping, lstm_dim=lstm_dim, bidirectional=False, num_layers=n_layers).to(device=device)
# model = feedBackLSTM(FPdataset.mapping, target_range=12, lstm_dim=lstm_dim).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), LR)
loss_func = nn.MSELoss()
print(model)
print(model._get_name())

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=99)

for epoch in range(50):
    model.train()
    train_loss = []
    for i, (input_ids, labels) in enumerate(tqdm(train_loader)):
        input_ids = input_ids.to(device=device)
        labels = labels.to(device=device)

        pred, _ = model(input_ids, labels) if type(model) == feedBackLSTM else model(input_ids)
        pred = pred[input_ids != FPdataset.mapping.pad_id]
        labels = labels[input_ids != FPdataset.mapping.pad_id]
        loss = loss_func(pred, labels.float())

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(),CLIP)
        optimizer.step()

        train_loss.append(loss.item())

    train_loss = sum(train_loss) / len(train_loss)

    with torch.no_grad():
        model.eval()
        test_loss = []
        test_acc = []
        test_acc1 = []
        best_acc = 0
        for i, (input_ids, labels) in enumerate(test_loader):
            input_ids = input_ids.to(device=device)
            labels = labels.to(device=device)

            pred, _ = model.decode(input_ids) if type(model) == feedBackLSTM else model(input_ids)
            pred = pred[input_ids != FPdataset.mapping.pad_id]
            labels = labels[input_ids != FPdataset.mapping.pad_id]

            loss = F.l1_loss(pred, labels.float())
            test_loss.append(loss.item())

            acc = (torch.round(pred).long() == labels).float().mean()
            test_acc.append(acc.item())

            accWithin1 = ((torch.round(pred) <= labels.float()+1) & (torch.round(pred) >= labels.float()-1) ).float().mean()
            test_acc1.append(accWithin1.item())

        test_loss = sum(test_loss) / len(test_loss)
        test_acc = sum(test_acc) / len(test_acc)
        test_acc1 = sum(test_acc1) / len(test_acc1)

    print('epoch %d:  train loss: %.4f  test loss: %.4f  test acc: %.4f  test acc (+-1): %.4f   LR: %f'%(epoch, train_loss, test_loss, test_acc, test_acc1, optimizer.param_groups[0]['lr']))
    lr_scheduler.step(test_loss)
    if test_acc1 > best_acc:
        best_acc = test_acc1
        torch.save((model, FPdataset.mapping), './FPmodels/model-%s-%s-%s.pt'%(model._get_name(), FPdataset.feature, 'biD' if model.bidirectional else 'uniD'))
print(BATCH_SIZE, LR, lstm_dim, type(model))