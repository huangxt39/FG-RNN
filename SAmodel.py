import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import math
from newModels import durationRNNCellv1, durationLSTMCellv1
from FPmodel import simpleLSTM, read_embedding


class baselineSA(nn.Module):
    def __init__(self, mapping, lstm_dim=100, dropout_rate=0.5, bidirectional=True, pretrained_embedding=None, rnn='lstm', num_layers=1):
        super().__init__()

        emb_dim=100
        
        embedding=nn.Embedding(len(mapping), emb_dim, padding_idx=0)
        if pretrained_embedding is not None:
            word2emb = pretrained_embedding

            """initialize with glove embedding"""
            for i in range(mapping.get_len()):
                word=mapping.get_word(i)
                if word in word2emb:
                    embedding.weight.data[i] = word2emb[word]

        self.embedding=embedding
        self.dropout_emb=nn.Dropout(dropout_rate)
        self.dropout_layer=nn.Dropout(0.25)
        if rnn == 'lstm':
            self.lstm=nn.LSTM(emb_dim, lstm_dim, num_layers=num_layers, batch_first=True, dropout=0.25, bidirectional=bidirectional)
        elif rnn == 'rnn':
            self.lstm=nn.RNN(emb_dim, lstm_dim, num_layers=num_layers, batch_first=True, dropout=0.25, bidirectional=bidirectional)

        self.hidden=nn.Sequential(
            nn.Linear(lstm_dim*2 if bidirectional else lstm_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim,2)
        )

        self.lstm_dim=lstm_dim
        self.emb_dim=emb_dim
        self.bidirectional = bidirectional
        self.rnn = rnn
        self.num_layers = num_layers

    def forward(self, input_ids, input_len, fix_duration):
        x = self.embedding(input_ids)
        x = self.dropout_emb(x)
        x, _ = self.lstm(x)

        input_len = input_len.unsqueeze(1).unsqueeze(2).expand(-1, -1, x.size(2))
        x = torch.gather(x, 1, input_len-1).squeeze(1)

        x = self.dropout_layer(x)
        x = self.hidden(x)
        return x

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


class durationLMAdapt(nn.Module):
    def __init__(self, rnn, vocab, embedding_dim, hidden_dim, n_hiddens, dropout_rate_emb, dropout_rate_layer, 
                tie_weights, FPmodel, scale_rate, use_same_emb, use_norm, adapt=True, num_layers=1, pretrained_embedding=None):        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_hiddens = n_hiddens
        self.rnn = rnn
        self.scale_rate = scale_rate
        self.use_same_emb = use_same_emb
        self.use_norm = use_norm
        self.num_layers = num_layers
        self.adapt = adapt

        if adapt:
            if FPmodel != None: 
                self.FPmodel = FPmodel
            elif rnn == 'lstmv1':
                self.FPmodel = simpleLSTM(vocab, lstm_dim=hidden_dim, bidirectional=False, pretrained_embedding=False, num_layers=num_layers)
            elif rnn == 'v1':
                self.FPmodel = simpleLSTM(vocab, lstm_dim=hidden_dim, bidirectional=False, pretrained_embedding=False, rnn='rnn', num_layers=num_layers)

        # if use_norm:
        #     self.norm_layer = nn.BatchNorm1d(1, affine=False)
            # self.norm_bias = nn.parameter.Parameter(torch.tensor(0.0))

        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        if pretrained_embedding is not None:
            word2emb = pretrained_embedding
            for i in range(len(vocab)):
                word = vocab.get_word(i).lower()
                if word in word2emb:
                    self.embedding.weight.data[i] = word2emb[word]
        if adapt and use_same_emb:
            self.FPmodel.word_emb = self.embedding

        if rnn == 'lstmv1':
            if num_layers == 1:
                self.rnn_cell = durationLSTMCellv1(embedding_dim, hidden_dim, n_hiddens, 12)
            else:
                self.rnn_cell_list = nn.ModuleList([durationLSTMCellv1(embedding_dim if i==0 else hidden_dim, hidden_dim, n_hiddens, 12) for i in range(num_layers)])
        elif rnn == 'v1':
            if num_layers == 1:
                self.rnn_cell = durationRNNCellv1(embedding_dim, hidden_dim, n_hiddens, 12)
            else:
                self.rnn_cell_list = nn.ModuleList([durationRNNCellv1(embedding_dim if i==0 else hidden_dim, hidden_dim, n_hiddens, 12) for i in range(num_layers)])
                
        if num_layers > 1:
            self.linear_list = nn.ModuleList([nn.Linear(n_hiddens*hidden_dim, hidden_dim) for i in range(num_layers-1)])
        
        self.dropout_emb = nn.Dropout(dropout_rate_emb)
        self.dropout_layer = nn.Dropout(dropout_rate_layer)
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim*n_hiddens, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 2)
        ) 
        

    def forward(self, src, input_len, fix_src):
        # src  (batch_size, seq_len)
        # hidden v1 (n_hiddens, batch_size, d_h)
        # fix_src  (batch_size, seq_len)
        hidden = self.init_hidden(len(src), src.device)

        if self.adapt:
            if self.use_same_emb:
                fix_src = src
            hidden, fix_hidden = hidden

            fix_pred, fix_hidden = self.fix_predict(fix_src, fix_hidden)
            # [2, 1 ... ]
            # [False, False,
            #  False, False,
            #  False, True,
            #  True,
            #  True,]
        else:
            fix_pred = fix_src

        batch_size = src.size(0)
        seq_len = src.size(1)

        embedding = self.dropout_emb(self.embedding(src))

        if self.num_layers == 1:
            output = []
            for i in range(seq_len):
                hidden = self.rnn_cell(embedding[:,i:i+1], hidden, fix_pred[:, i], adapt=self.adapt, scale_rate=self.scale_rate)
                if self.rnn == 'lstmv1':
                    output.append(hidden[0].transpose(0,1).contiguous().view(batch_size, -1).unsqueeze(1))
                elif self.rnn == 'v1':
                    output.append(hidden.transpose(0,1).contiguous().view(batch_size, -1).unsqueeze(1))
            output = torch.cat(output, dim=1)   # batch_size, seq_len, d_h*n_hidden
            output = self.dropout_layer(output) 
        else:
            x = embedding.transpose(0,1).unsqueeze(1).expand(-1, self.n_hiddens, -1, -1).contiguous()
            for l in range(self.num_layers):
                output = []
                h = hidden[l]
                for i in range(seq_len):
                    h = self.rnn_cell_list[l](x[i:i+1], h, fix_pred[:, i], adapt=self.adapt, scale_rate=self.scale_rate)
                    if self.rnn == 'lstmv1':
                        output.append(h[0].unsqueeze(0))
                    elif self.rnn == 'v1':
                        output.append(h.unsqueeze(0))
                x = torch.cat(output, dim=0)   # seq_len, n_hidden, batch_size, d_h
                x = self.dropout_layer(x)
                hidden[l] = h

                if l < self.num_layers-1:
                    x = self.linear_list[l](x.transpose(1,2).contiguous().view(seq_len, batch_size, -1)) # seq_len, batch_size, d_h
                    x = x.unsqueeze(1).expand(-1, self.n_hiddens, -1, -1).contiguous()
                
            output = x.permute(2, 0, 1, 3).contiguous().view(batch_size, seq_len, -1)
        
        input_len = input_len.unsqueeze(1).unsqueeze(2).expand(-1, -1, output.size(2))
        output = torch.gather(output, 1, input_len-1).squeeze(1)

        prediction = self.fc(output)
        if self.adapt:
            hidden = hidden, fix_hidden
        return prediction

    def fix_predict(self, fix_src, fix_hidden):
        fix_pred, fix_hidden = self.FPmodel(fix_src, fix_hidden)    # fix_pred  (batch_size, seq_len)
        if self.use_norm:
            # fix_pred = self.norm_layer(fix_pred.unsqueeze(1)).squeeze(1)
            fix_pred = F.layer_norm(fix_pred, [fix_pred.size(1)])
            fix_pred = (fix_pred + 1.96) / 3.92 * 12        # *10     # test 10 epochs  **3
        return fix_pred, fix_hidden

    def init_hidden(self, batch_size, device):
        if self.num_layers == 1:
            cell_hidden = self.rnn_cell.init_hidden(batch_size, device)
        else:
            cell_hidden = []
            for l in self.rnn_cell_list:
                cell_hidden.append(l.init_hidden(batch_size, device))
        if self.adapt:
            return (cell_hidden, self.FPmodel.init_hidden(batch_size, device))
        else:
            return cell_hidden

    def detach_hidden(self, hidden):
        if self.adapt:
            hidden, fix_hidden = hidden
        if self.num_layers == 1:
            if self.rnn == 'lstmv1':
                new_hidden = (hidden[0].detach(), hidden[1].detach())
            elif self.rnn == 'v1':
                new_hidden = hidden.detach()             
        else:
            if self.rnn == 'lstmv1':
                new_hidden = [(h[0].detach(), h[1].detach()) for h in hidden]
            elif self.rnn == 'v1':
                new_hidden = [h.detach() for h in hidden]
        if self.adapt:
            return ( new_hidden, self.FPmodel.detach_hidden(fix_hidden) )
        else:
            return new_hidden


class durationLMLayer(nn.Module):
    def __init__(self, rnn, vocab, embedding_dim, hidden_dim, dropout_rate_emb, dropout_rate_layer, 
                tie_weights, FPmodel, scale_rate, use_same_emb, use_norm, adapt=True, num_layers=1, pretrained_embedding=True):        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.rnn = rnn
        self.scale_rate = scale_rate
        self.use_same_emb = use_same_emb
        self.use_norm = use_norm
        self.num_layers = num_layers
        self.adapt = adapt
        self.rescale_rate = 12 // self.num_layers

        assert num_layers > 1

        if adapt:
            if FPmodel != None: 
                self.FPmodel = FPmodel
            elif rnn == 'lstm':
                self.FPmodel = simpleLSTM(vocab, lstm_dim=hidden_dim, bidirectional=False, pretrained_embedding=False, num_layers=num_layers)
            elif rnn == 'rnn':
                self.FPmodel = simpleLSTM(vocab, lstm_dim=hidden_dim, bidirectional=False, pretrained_embedding=False, rnn='rnn', num_layers=num_layers)

        # if use_norm:
        #     self.norm_layer = nn.BatchNorm1d(1, affine=False)
            # self.norm_bias = nn.parameter.Parameter(torch.tensor(0.0))

        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        if pretrained_embedding:
            word2emb = read_embedding()
            for i in range(len(vocab)):
                word = vocab.lookup_token(i).lower()
                if word in word2emb:
                    self.embedding.weight.data[i] = word2emb[word]
        if adapt and use_same_emb:
            self.FPmodel.word_emb = self.embedding

        if rnn == 'lstm':
            self.rnn_cell_list = nn.ModuleList([nn.LSTMCell(embedding_dim if i==0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        elif rnn == 'rnn':
            self.rnn_cell_list = nn.ModuleList([nn.RNNCell(embedding_dim if i==0 else hidden_dim, hidden_dim) for i in range(num_layers)])
                
        self.dropout_emb = nn.Dropout(dropout_rate_emb)
        self.dropout_layer = nn.Dropout(dropout_rate_layer)
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim*num_layers, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 2)
        ) 
         

    def forward(self, src, hidden, fix_src):
        # src  (batch_size, seq_len)
        # hidden v1 (n_hiddens, batch_size, d_h)
        # fix_src  (batch_size, seq_len)
        if self.adapt:
            if self.use_same_emb:
                fix_src = src
            hidden, fix_hidden = hidden

            fix_pred, fix_hidden = self.fix_predict(fix_src, fix_hidden)
            # [2, 1 ... ]
            # [False, False,
            #  False, False,
            #  False, True,
            #  True,
            #  True,]
        else:
            fix_pred = fix_src

        batch_size = src.size(0)
        seq_len = src.size(1)

        embedding = self.dropout_emb(self.embedding(src))

        x = embedding.transpose(0,1)
        output = []
        for l in range(self.num_layers):
            l_output = []
            h = hidden[l]
            for i in range(seq_len):
                if self.rnn == 'lstm':
                    h_old, c_old = h
                elif self.rnn == 'rnn':
                    h_old = h
                
                h = self.rnn_cell_list[l](x[i], h)
                if self.adapt:
                    if l == 0:
                        mask = torch.zeros((batch_size, 1), dtype=x.dtype, device=x.device)
                    else:
                        mask = l -  (fix_pred[:, i:i+1] / self.rescale_rate) # - 0.5 - for D16 D17 E17
                        mask = torch.sigmoid(mask * self.scale_rate)
                else:
                    mask = l > torch.div(fix_pred[:, i:i+1], self.rescale_rate, rounding_mode='floor')
                    mask = mask.float()

                if self.rnn == 'lstm':
                    hx, cx = h
                    cx = cx * (1 - mask) + c_old * mask
                    hx = hx * (1 - mask) + h_old * mask
                    h = (hx, cx)
                    l_output.append(h[0].unsqueeze(0))
                elif self.rnn == 'rnn':
                    h = h * (1 - mask) + h_old * mask
                    l_output.append(h.unsqueeze(0))
            x = torch.cat(l_output, dim=0)   
            x = self.dropout_layer(x)
            hidden[l] = h

            output.append(x.unsqueeze(0))   
        output = torch.cat(output, dim=0)    # n_layer, seq_len, batch_size, h_dim
        output = output.transpose(0,2).contiguous().view(batch_size, seq_len, -1)
        
        
        prediction = self.fc(output)
        if self.adapt:
            hidden = hidden, fix_hidden
        return prediction, hidden

    def fix_predict(self, fix_src, fix_hidden):
        fix_pred, fix_hidden = self.FPmodel(fix_src, fix_hidden)    # fix_pred  (batch_size, seq_len)
        if self.use_norm:
            # fix_pred = self.norm_layer(fix_pred.unsqueeze(1)).squeeze(1)
            fix_pred = F.layer_norm(fix_pred, [fix_pred.size(1)])
            fix_pred = (fix_pred + 1.96) / 3.92 * 12        # *10     # test 10 epochs  **3
        return fix_pred, fix_hidden

    def init_hidden(self, batch_size, device):
        if self.rnn == 'lstm':
            cell_hidden = [ (torch.zeros(batch_size, self.hidden_dim).to(device), \
                    torch.zeros(batch_size, self.hidden_dim).to(device)) for i in range(self.num_layers)]
        elif self.rnn == 'rnn':
            cell_hidden = [ torch.zeros(batch_size, self.hidden_dim).to(device) for i in range(self.num_layers)]
        if self.adapt:
            return (cell_hidden, self.FPmodel.init_hidden(batch_size, device))
        else:
            return cell_hidden

    def detach_hidden(self, hidden):
        if self.adapt:
            hidden, fix_hidden = hidden   
        if self.rnn == 'lstm':
            new_hidden = [(h[0].detach(), h[1].detach()) for h in hidden]
        elif self.rnn == 'rnn':
            new_hidden = [h.detach() for h in hidden]
        if self.adapt:
            return ( new_hidden, self.FPmodel.detach_hidden(fix_hidden) )
        else:
            return new_hidden
