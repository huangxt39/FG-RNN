import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

def read_embedding():
    print('loading embedding')
    s_time = time.time()
    f=open('wordEmbedding/glove.6B.100d.txt',encoding="utf-8")
    line=f.readline()
    word2emb={}
    while line:
        line=line.split()
        word2emb[line[0]]=torch.from_numpy(np.array(line[1:],dtype=np.dtype(str)).astype(np.dtype(float)))
        line=f.readline()
    print('embedding loading time: ', time.time()-s_time)
    return word2emb

class simpleLSTM(nn.Module):
    def __init__(self, mapping, lstm_dim=100, dropout_rate=0.5, bidirectional=True, pretrained_embedding=True, rnn='lstm', num_layers=1):
        super().__init__()

        emb_dim=100
        
        word_emb=nn.Embedding(len(mapping), emb_dim, padding_idx=0)
        if pretrained_embedding:
            """load embedding"""
            word2emb = read_embedding()

            """initialize with glove embedding"""
            for i in range(mapping.get_len()):
                word=mapping.get_word(i)
                if word in word2emb:
                    word_emb.weight.data[i] = word2emb[word]

        self.word_emb=word_emb
        self.dropout=nn.Dropout(dropout_rate)
        if rnn == 'lstm':
            self.lstm=nn.LSTM(emb_dim, lstm_dim, num_layers=num_layers, batch_first=True, dropout=0.0, bidirectional=bidirectional)
        elif rnn == 'rnn':
            self.lstm=nn.RNN(emb_dim, lstm_dim, num_layers=num_layers, batch_first=True, dropout=0.0, bidirectional=bidirectional)

        self.hidden=nn.Sequential(
            nn.Linear(lstm_dim*2 if bidirectional else lstm_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim,1)
        )

        # self.a = nn.parameter.Parameter(torch.tensor(1.0))
        # self.b = nn.parameter.Parameter(torch.tensor(0.0))

        self.lstm_dim=lstm_dim
        self.emb_dim=emb_dim
        self.bidirectional = bidirectional
        self.rnn = rnn
        self.num_layers = num_layers

    def forward(self, input_ids, h=None):
        x = self.word_emb(input_ids)
        x = self.dropout(x)
        x, h = self.lstm(x, h)
        x = self.dropout(x)
        x = self.hidden(x).squeeze(2) #* self.a + self.b    # batch_size, sequence_len
        return x, h

    def init_hidden(self, batch_size, device):
        num_directions = 2 if self.bidirectional else 1
        if self.rnn == 'lstm':
            return (torch.zeros(self.num_layers*num_directions, batch_size, self.lstm_dim).to(device), \
                torch.zeros(self.num_layers*num_directions, batch_size, self.lstm_dim).to(device))
        elif self.rnn == 'rnn':
            return torch.zeros(self.num_layers*num_directions, batch_size, self.lstm_dim).to(device)
    
    def detach_hidden(self, hidden):
        if self.rnn == 'lstm':
            return (hidden[0].detach(), hidden[1].detach())
        elif self.rnn == 'rnn':
            return hidden.detach()
        

class feedBackLSTM(nn.Module):
    def __init__(self, mapping, target_range, lstm_dim=100, dropout_rate=0.5):
        super().__init__()

        emb_dim=100
        
        """load embedding"""
        word2emb = read_embedding()

        """initialize with glove embedding"""
        word_emb=nn.Embedding(mapping.get_len(), emb_dim, padding_idx=0)
        for i in range(mapping.get_len()):
            word=mapping.get_word(i)
            word_emb.weight.data[i] = word2emb.get(word, 0)

        self.word_emb=word_emb
        self.dropout=nn.Dropout(dropout_rate)
        self.lstm=nn.LSTM(emb_dim + target_range, lstm_dim, batch_first=True, bidirectional=False)
        self.hidden=nn.Sequential(
            nn.Linear(lstm_dim,lstm_dim),
            nn.Tanh(),
            nn.Linear(lstm_dim,1)
        )

        self.lstm_dim = lstm_dim
        self.emb_dim = emb_dim
        self.target_range = target_range
        self.mapping = mapping

    def forward(self, input_ids, labels):
        labels = F.one_hot(labels, num_classes=self.target_range)
        past_labels = labels[:, :-1].clone()
        past_labels = torch.cat( (torch.zeros(len(input_ids), 1, self.target_range, dtype=torch.long, device=past_labels.device), past_labels), dim=1).float()
        past_labels.masked_fill_((input_ids==self.mapping.pad_id).unsqueeze(2), 0)

        x = self.word_emb(input_ids)

        x = self.dropout(x)
        x = torch.cat((past_labels, x), dim=2)
        x, h = self.lstm(x)
        x = self.hidden(x).squeeze(2)    # batch_size, sequence_len
        return x, h

    def decode(self, input_ids):
        x = self.word_emb(input_ids)
        past_label = torch.zeros(len(input_ids), 1, self.target_range, device=x.device)
        h = (torch.zeros(1, len(input_ids), self.lstm_dim, device=x.device), \
            torch.zeros(1, len(input_ids), self.lstm_dim, device=x.device) )
        pred = []
        for i in range(input_ids.size(1)):
            t = torch.cat( (past_label, x[:, i:i+1]), dim=2)
            t, h = self.lstm(t, h)
            t = self.hidden(t).squeeze()
            pred.append(t.unsqueeze(1))
            t = torch.clamp(torch.round(t).long(), min=0, max=self.target_range-1)

            past_label = F.one_hot(t, num_classes=self.target_range).unsqueeze(1).float()

        return torch.cat(pred, dim=1)
