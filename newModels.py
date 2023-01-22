import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from FPmodel import read_embedding, simpleLSTM

class durationRNNCellv1(nn.Module):
    def __init__(self, input_size, hidden_size, n_hiddens, max_duration):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hiddens = n_hiddens
        self.rescale_rate = max_duration // n_hiddens

        self.weight_ih = nn.parameter.Parameter(torch.zeros(n_hiddens, input_size, hidden_size))
        self.weight_hh = nn.parameter.Parameter(torch.zeros(n_hiddens, hidden_size, hidden_size))
        self.bias_ih = nn.parameter.Parameter(torch.zeros(n_hiddens, hidden_size))
        self.bias_hh = nn.parameter.Parameter(torch.zeros(n_hiddens, hidden_size))
        self.non_linear = nn.Tanh()

        self.init_weights()

    def forward(self, input_x, hidden, fix_duration, adapt=False, scale_rate=2):
        # input_x (batch_size, 1, d_i)  d_i: input_size 
        # hidden (n_hiddens, batch_size, d_h)
        # fix_duration (batch_size,)
        
        if input_x.dim() == 3:
            batch_size = input_x.size(0)
            input_x = input_x.expand(-1, self.n_hiddens, -1).transpose(0,1).contiguous() # (n_hidden, batch_size, d_i)
        elif input_x.dim() == 4:
            batch_size = input_x.size(2)
            input_x = input_x.squeeze(0)

        if adapt:
            mask = (torch.arange(start=1, end=self.n_hiddens, device=input_x.device).unsqueeze(1).expand(-1, batch_size) - 0.5 - \
                            (fix_duration / self.rescale_rate).unsqueeze(0)).unsqueeze(2)  # (n_hidden, batch_size, 1)
            mask = torch.sigmoid(mask * scale_rate)
            mask = torch.cat([torch.zeros((1, batch_size, 1), dtype=mask.dtype, device=mask.device), mask], dim=0)
        else:
            mask = (torch.arange(self.n_hiddens, device=input_x.device).unsqueeze(1).expand(-1, batch_size) > \
                            torch.div(fix_duration, self.rescale_rate, rounding_mode='floor').unsqueeze(0)).unsqueeze(2)  # (n_hidden, batch_size, 1)
            mask = mask.float()
        
        input_x = torch.bmm(input_x, self.weight_ih)  # (n_hidden, batch_size, d_h)
        input_x += self.bias_ih.unsqueeze(1).expand(-1, batch_size, -1)

        hidden_old = hidden.clone()
        hidden = torch.bmm(hidden, self.weight_hh)  # (n_hidden, batch_size, d_h)
        hidden += self.bias_hh.unsqueeze(1).expand(-1, batch_size, -1)

        hidden = self.non_linear(input_x + hidden)  # relu and tanh : 0 -> 0 so we don't need mask again

        hidden = hidden * (1 - mask) + hidden_old * mask

        return hidden

    
    def forward_adapt(self, input_x, hidden, fix_duration, scale_rate=2):
        # input_x (batch_size, 1, d_i)  d_i: input_size / (1, n_hidden, batch_size, d_i)
        # hidden (n_hiddens, batch_size, d_h)
        # fix_duration (batch_size,)
        if input_x.dim() == 3:
            batch_size = input_x.size(0)
            input_x = input_x.expand(-1, self.n_hiddens, -1).transpose(0,1).contiguous() # (n_hidden, batch_size, d_i)
        elif input_x.dim() == 4:
            batch_size = input_x.size(2)
            input_x = input_x.squeeze(0)
        mask = (torch.arange(start=1, end=self.n_hiddens, device=input_x.device).unsqueeze(1).expand(-1, batch_size) - 0.5 - \
                            (fix_duration / self.rescale_rate).unsqueeze(0)).unsqueeze(2)  # (n_hidden, batch_size, 1)
        mask = torch.sigmoid(mask * scale_rate)
        mask = torch.cat([torch.zeros((1, batch_size, 1), dtype=mask.dtype, device=mask.device), mask], dim=0)
        

        input_x = torch.bmm(input_x, self.weight_ih)  # (n_hidden, batch_size, d_h)
        input_x += self.bias_ih.unsqueeze(1).expand(-1, batch_size, -1)

        hidden_old = hidden.clone()
        hidden = torch.bmm(hidden, self.weight_hh)  # (n_hidden, batch_size, d_h)
        hidden += self.bias_hh.unsqueeze(1).expand(-1, batch_size, -1)

        hidden = self.non_linear(input_x + hidden)  

        hidden = hidden * (1 - mask) + hidden_old * mask

        return hidden

    def init_weights(self):
        k = math.sqrt(1/self.hidden_size)
        nn.init.uniform_(self.weight_ih.data, -k, k)
        nn.init.uniform_(self.weight_hh.data, -k, k)
        nn.init.uniform_(self.bias_ih.data, -k, k)
        nn.init.uniform_(self.bias_hh.data, -k, k)

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_hiddens, batch_size, self.hidden_size).to(device)


class durationRNNCellv2(nn.Module):
    def __init__(self, input_size, hidden_size, n_hiddens, max_duration):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hiddens = n_hiddens
        self.rescale_rate = max_duration // n_hiddens

        self.weight_ih = nn.parameter.Parameter(torch.zeros(input_size, n_hiddens * hidden_size))
        self.weight_hh = nn.parameter.Parameter(torch.zeros(n_hiddens * hidden_size, n_hiddens * hidden_size))
        self.bias_ih = nn.parameter.Parameter(torch.zeros(n_hiddens * hidden_size,))
        self.bias_hh = nn.parameter.Parameter(torch.zeros(n_hiddens * hidden_size,))
        self.non_linear = nn.Tanh()

        self.init_weights()

    def forward(self, input_x, hidden, fix_duration):
        # input_x (batch_size, 1, d_i)  d_i: input_size
        # hidden (batch_size, n_hiddens * d_h)
        # fix_duration (batch_size,)

        batch_size = input_x.size(0)

        input_x = torch.mm(input_x.squeeze(), self.weight_ih) + self.bias_ih.unsqueeze(0)
        mask = torch.arange(self.n_hiddens, device=input_x.device).unsqueeze(0).expand(batch_size, -1) > \
                            torch.div(fix_duration, self.rescale_rate, rounding_mode='floor').unsqueeze(1)  # (batch_size, n_hiddens)
        mask = mask.unsqueeze(2).expand(-1, -1, self.hidden_size).contiguous().view(batch_size, self.n_hiddens*self.hidden_size)     # (batch_size, n_hiddens*d_h)
        
        input_x = input_x.masked_fill(mask, 0)  # (batch_size, n_hiddens*d_h)

        hidden_old = hidden.clone()
        # hidden = torch.mm(hidden.masked_fill(mask, 0), self.weight_hh) + self.bias_hh.unsqueeze(0)
        hidden = torch.mm(hidden, self.weight_hh) + self.bias_hh.unsqueeze(0)
        hidden = hidden.masked_fill(mask, 0)    # (batch_size, n_hiddens*d_h)

        hidden = self.non_linear(input_x + hidden)
        hidden = hidden + hidden_old * mask.float()

        return hidden

    def init_weights(self):
        k = math.sqrt(1/(self.n_hiddens * self.hidden_size))
        nn.init.uniform_(self.weight_ih.data, -k, k)
        nn.init.uniform_(self.weight_hh.data, -k, k)
        nn.init.uniform_(self.bias_ih.data, -k, k)
        nn.init.uniform_(self.bias_hh.data, -k, k)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.n_hiddens * self.hidden_size).to(device)


class durationRNNCellv4(nn.Module):
    def __init__(self, input_size, hidden_size, n_hiddens, max_duration):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hiddens = n_hiddens
        self.rescale_rate = max_duration // n_hiddens

        self.weight_ih = nn.parameter.Parameter(torch.zeros(1, input_size, hidden_size))
        self.weight_hh = nn.parameter.Parameter(torch.zeros(1, hidden_size, hidden_size))
        self.bias_ih = nn.parameter.Parameter(torch.zeros(1, hidden_size))
        self.bias_hh = nn.parameter.Parameter(torch.zeros(1, hidden_size))
        self.non_linear = nn.Tanh()

        self.init_weights()

    def forward(self, input_x, hidden, fix_duration):
        # input_x (batch_size, 1, d_i)  d_i: input_size
        # hidden (n_hiddens, batch_size, d_h)
        # fix_duration (batch_size,)
        # exp with contiguous, exp with expand, masked_fill, see how grad accumulate : conclusion: it acts in a right way
        batch_size = input_x.size(0)
        input_x = input_x.transpose(0,1).contiguous() # (1, batch_size, d_i)
        mask = (torch.arange(self.n_hiddens, device=input_x.device).unsqueeze(1).expand(-1, batch_size) > \
                            torch.div(fix_duration, self.rescale_rate, rounding_mode='floor').unsqueeze(0)).unsqueeze(2)  # (n_hidden, batch_size, 1)
        
        # masked_fill can be move to later
        input_x = torch.bmm(input_x, self.weight_ih)  # (1, batch_size, d_h)
        input_x += self.bias_ih.unsqueeze(1).expand(-1, batch_size, -1).contiguous()

        hidden_old = hidden.clone()
        hidden = torch.bmm(hidden.masked_fill(mask, 0), self.weight_hh.expand(self.n_hiddens, -1, -1))  # (n_hidden, batch_size, d_h)
        hidden += self.bias_hh.unsqueeze(1).expand(self.n_hiddens, batch_size, -1).contiguous().masked_fill(mask, 0)

        hidden = self.non_linear(input_x + hidden)  

        # hidden += hidden_old.masked_fill(~mask, 0)  # problematic in backward()
        hidden = hidden * (1-mask.float()) + hidden_old * mask.float()

        return hidden

    def init_weights(self):
        k = math.sqrt(1/self.hidden_size)
        nn.init.uniform_(self.weight_ih.data, -k, k)
        nn.init.uniform_(self.weight_hh.data, -k, k)
        nn.init.uniform_(self.bias_ih.data, -k, k)
        nn.init.uniform_(self.bias_hh.data, -k, k)

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_hiddens, batch_size, self.hidden_size).to(device)

class durationLSTMCellv1(nn.Module):
    def __init__(self, input_size, hidden_size, n_hiddens, max_duration):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hiddens = n_hiddens
        self.rescale_rate = max_duration // n_hiddens

        self.weight_ih_ifog = nn.parameter.Parameter(torch.zeros(n_hiddens, input_size + hidden_size, hidden_size * 4))
        self.bias_i_ifog = nn.parameter.Parameter(torch.zeros(n_hiddens, hidden_size * 4))
        self.bias_h_ifog = nn.parameter.Parameter(torch.zeros(n_hiddens, hidden_size * 4))  # don't really need two bias, but for the fairness when comparing number of params
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(self, input_x, hidden, fix_duration, adapt=False, scale_rate=2):
        # input_x (batch_size, 1, d_i)  d_i: input_size
        # hidden h: (n_hiddens, batch_size, d_h) c: (n_hiddens, batch_size, d_h)
        # fix_duration (batch_size,)

        if input_x.dim() == 3:
            batch_size = input_x.size(0)
            input_x = input_x.expand(-1, self.n_hiddens, -1).transpose(0,1).contiguous() # (n_hidden, batch_size, d_i)
        elif input_x.dim() == 4:
            batch_size = input_x.size(2)
            input_x = input_x.squeeze(0)
        
        if adapt:
            mask = (torch.arange(start=1, end=self.n_hiddens, device=input_x.device).unsqueeze(1).expand(-1, batch_size) - 0.5 - \
                            (fix_duration / self.rescale_rate).unsqueeze(0)).unsqueeze(2)  # (n_hidden, batch_size, 1)
            mask = torch.sigmoid(mask * scale_rate)
            mask = torch.cat([torch.zeros((1, batch_size, 1), dtype=mask.dtype, device=mask.device), mask], dim=0)
        else:
            mask = (torch.arange(self.n_hiddens, device=input_x.device).unsqueeze(1).expand(-1, batch_size) > \
                            torch.div(fix_duration, self.rescale_rate, rounding_mode='floor').unsqueeze(0)).unsqueeze(2)  # (n_hidden, batch_size, 1)
            mask = mask.float()
        
        h, c = hidden
        h_old, c_old = h.clone(), c.clone()
        combined = torch.cat((input_x, h), dim=2)
        h = torch.bmm(combined, self.weight_ih_ifog)  # (n_hidden, batch_size, d_h*4)
        h = h + (self.bias_i_ifog + self.bias_h_ifog).unsqueeze(1).expand(-1, batch_size, -1)

        d_h = self.hidden_size
        gate_ifo = self.sigmoid(h[:,:,:d_h*3])
        input_node = self.tanh(h[:,:,d_h*3:])

        # c = gate_ifo[:,:,d_h:d_h*2] * c + gate_ifo[:,:,:d_h] * input_node
        # h = gate_ifo[:,:,d_h*2:] * self.tanh(c)
        
        # c = c * (1 - mask) + c_old * mask
        # h = h * (1 - mask) + h_old * mask

        c = gate_ifo[:,:,d_h:d_h*2] * c + gate_ifo[:,:,:d_h] * input_node
        c = c * (1 - mask) + c_old * mask
        h = gate_ifo[:,:,d_h*2:] * self.tanh(c)

        return (h, c)

    def forward_adapt(self, input_x, hidden, fix_duration, scale_rate=2):
        # input_x (batch_size, 1, d_i)  d_i: input_size  / (1, n_hidden, batch_size, d_i)
        # hidden h: (n_hiddens, batch_size, d_h) c: (n_hiddens, batch_size, d_h)
        # fix_duration (batch_size,)
        if input_x.dim() == 3:
            batch_size = input_x.size(0)
            input_x = input_x.expand(-1, self.n_hiddens, -1).transpose(0,1).contiguous() # (n_hidden, batch_size, d_i)
        elif input_x.dim() == 4:
            batch_size = input_x.size(2)
            input_x = input_x.squeeze(0)
        mask = (torch.arange(start=1, end=self.n_hiddens, device=input_x.device).unsqueeze(1).expand(-1, batch_size) - 0.5 - \
                            (fix_duration / self.rescale_rate).unsqueeze(0)).unsqueeze(2)  # (n_hidden, batch_size, 1)
        mask = self.sigmoid(mask * scale_rate)
        mask = torch.cat([torch.zeros((1, batch_size, 1), dtype=mask.dtype, device=mask.device), mask], dim=0)
        
        h, c = hidden
        h_old, c_old = h.clone(), c.clone()
        combined = torch.cat((input_x, h), dim=2)
        h = torch.bmm(combined, self.weight_ih_ifog)  # (n_hidden, batch_size, d_h*4)
        h = h + (self.bias_i_ifog + self.bias_h_ifog).unsqueeze(1).expand(-1, batch_size, -1).contiguous()

        d_h = self.hidden_size
        gate_ifo = self.sigmoid(h[:,:,:d_h*3])
        input_node = self.tanh(h[:,:,d_h*3:])

        # c = gate_ifo[:,:,d_h:d_h*2] * c + gate_ifo[:,:,:d_h] * input_node
        # h = gate_ifo[:,:,d_h*2:] * self.tanh(c)

        # c = c * (1 - mask) + c_old * mask
        # h = h * (1 - mask) + h_old * mask

        c = gate_ifo[:,:,d_h:d_h*2] * c + gate_ifo[:,:,:d_h] * input_node
        c = c * (1 - mask) + c_old * mask
        h = gate_ifo[:,:,d_h*2:] * self.tanh(c)

        return (h, c)

    def init_weights(self):
        k = math.sqrt(1/self.hidden_size)
        nn.init.uniform_(self.weight_ih_ifog.data, -k, k)
        nn.init.uniform_(self.bias_i_ifog.data, -k, k)
        nn.init.uniform_(self.bias_h_ifog.data, -k, k)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.n_hiddens, batch_size, self.hidden_size).to(device), \
                torch.zeros(self.n_hiddens, batch_size, self.hidden_size).to(device))


class durationLSTMCellv4(nn.Module):
    def __init__(self, input_size, hidden_size, n_hiddens, max_duration):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hiddens = n_hiddens
        self.rescale_rate = max_duration // n_hiddens

        self.weight_i_ifog = nn.parameter.Parameter(torch.zeros(1, input_size, hidden_size * 4))
        self.weight_h_ifog = nn.parameter.Parameter(torch.zeros(1, hidden_size, hidden_size * 4))
        self.bias_i_ifog = nn.parameter.Parameter(torch.zeros(1, hidden_size * 4))
        self.bias_h_ifog = nn.parameter.Parameter(torch.zeros(1, hidden_size * 4))  # don't really need two bias, but for the fairness when comparing number of params
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(self, input_x, hidden, fix_duration):
        # input_x (batch_size, 1, d_i)  d_i: input_size
        # hidden h: (n_hiddens, batch_size, d_h) c: (n_hiddens, batch_size, d_h)
        # fix_duration (batch_size,)

        batch_size = input_x.size(0)
        input_x = input_x.transpose(0,1).contiguous() # (1, batch_size, d_i)
        mask = (torch.arange(self.n_hiddens, device=input_x.device).unsqueeze(1).expand(-1, batch_size) > \
                            torch.div(fix_duration, self.rescale_rate, rounding_mode='floor').unsqueeze(0)).unsqueeze(2)  # (n_hidden, batch_size, 1)
        
        # masked_fill can be move to later
        input_x = torch.bmm(input_x, self.weight_i_ifog)    # (1, batch_size, d_h*4)
        input_x += self.bias_i_ifog.unsqueeze(1).expand(-1, batch_size, -1).contiguous()

        h, c = hidden
        h_old, c_old = h.clone(), c.clone()

        h = torch.bmm(h.masked_fill(mask, 0), self.weight_h_ifog.expand(self.n_hiddens, -1, -1))  # (n_hidden, batch_size, d_h*4)
        h += self.bias_h_ifog.unsqueeze(1).expand(self.n_hiddens, batch_size, -1).contiguous().masked_fill(mask, 0)

        h = h + input_x
        d_h = self.hidden_size
        gate_ifo = self.sigmoid(h[:,:,:d_h*3])
        input_node = self.tanh(h[:,:,d_h*3:])

        c = gate_ifo[:,:,d_h:d_h*2] * c + gate_ifo[:,:,:d_h] * input_node
        h = gate_ifo[:,:,d_h*2:] * self.tanh(c)

        mask = mask.float()
        c = c * (1 - mask) + c_old * mask
        h = h * (1 - mask) + h_old * mask

        return (h, c)

    def init_weights(self):
        k = math.sqrt(1/self.hidden_size)
        nn.init.uniform_(self.weight_i_ifog.data, -k, k)
        nn.init.uniform_(self.weight_h_ifog.data, -k, k)
        nn.init.uniform_(self.bias_i_ifog.data, -k, k)
        nn.init.uniform_(self.bias_h_ifog.data, -k, k)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.n_hiddens, batch_size, self.hidden_size).to(device), \
                torch.zeros(self.n_hiddens, batch_size, self.hidden_size).to(device))



class durationLMAdapt(nn.Module):
    def __init__(self, rnn, vocab, embedding_dim, hidden_dim, n_hiddens, dropout_rate_emb, dropout_rate_layer, 
                tie_weights, FPmodel, scale_rate, use_same_emb, use_norm, adapt=True, num_layers=1, pretrained_embedding=True):        
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
        if pretrained_embedding:
            word2emb = read_embedding()
            for i in range(len(vocab)):
                word = vocab.lookup_token(i).lower()
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
            nn.Linear(embedding_dim, len(vocab))
        ) 
        
        if tie_weights:
            self.fc[2].weight = self.embedding.weight  

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
                self.FPmodel = simpleLSTM(vocab, lstm_dim=hidden_dim//3, bidirectional=False, pretrained_embedding=False, num_layers=num_layers)
            elif rnn == 'rnn':
                self.FPmodel = simpleLSTM(vocab, lstm_dim=hidden_dim//3, bidirectional=False, pretrained_embedding=False, rnn='rnn', num_layers=num_layers)

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
            nn.Linear(embedding_dim, len(vocab))
        ) 
        
        if tie_weights:
            self.fc[2].weight = self.embedding.weight  

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


class durationLMLayer2(nn.Module):
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

        self.linear = nn.Linear(embedding_dim, hidden_dim)

        if rnn == 'lstm':
            self.rnn_cell_list = nn.ModuleList([nn.LSTMCell(hidden_dim, hidden_dim) for i in range(num_layers)])
        elif rnn == 'rnn':
            self.rnn_cell_list = nn.ModuleList([nn.RNNCell(hidden_dim, hidden_dim) for i in range(num_layers)])
                
        self.dropout_emb = nn.Dropout(dropout_rate_emb)
        self.dropout_layer = nn.Dropout(dropout_rate_layer)
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, len(vocab))
        ) 
        
        if tie_weights:
            self.fc[2].weight = self.embedding.weight  

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

        input_x = self.linear(embedding.transpose(0,1))
        x = torch.zeros(input_x.size(), dtype=input_x.dtype, device=input_x.device)
        for l in range(self.num_layers):
            l_output = []
            h = hidden[l]
            for i in range(seq_len):
                if self.rnn == 'lstm':
                    h_old, c_old = h
                elif self.rnn == 'rnn':
                    h_old = h
                
                
                if self.adapt:
                    alpha = ((11 - fix_pred[:, i:i+1] ) / self.rescale_rate) - l
                    mask = torch.sigmoid(alpha * self.scale_rate)
                    input_gate = torch.sigmoid((alpha + 1) * self.scale_rate) - mask
                        
                else:
                    raise RuntimeError
                    mask = l > torch.div(fix_pred[:, i:i+1], self.rescale_rate, rounding_mode='floor')
                    mask = mask.float()

                input_ = x[i] * (1 - input_gate) + input_x[i] * input_gate
                h = self.rnn_cell_list[l](input_, h)
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

              
        output = x.transpose(0,1)   # , batch_size,seq_len, h_dim
        
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

class durationLM(nn.Module):
    def __init__(self, rnn, vocab, embedding_dim, hidden_dim, n_hiddens, dropout_rate_emb, dropout_rate_layer, 
                tie_weights, pretrained_embedding=True):        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_hiddens = n_hiddens
        self.rnn = rnn

        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        if pretrained_embedding:
            word2emb = read_embedding()
            for i in range(len(vocab)):
                word = vocab.lookup_token(i).lower()
                if word in word2emb:
                    self.embedding.weight.data[i] = word2emb[word]

        if rnn == 'v1':
            self.rnn_cell = durationRNNCellv1(embedding_dim, hidden_dim, n_hiddens, 12)
        elif rnn == 'v2':
            self.rnn_cell = durationRNNCellv2(embedding_dim, hidden_dim, n_hiddens, 12)
        elif rnn == 'v4':
            self.rnn_cell = durationRNNCellv4(embedding_dim, hidden_dim, n_hiddens, 12)
        elif rnn == 'lstmv1':
            self.rnn_cell = durationLSTMCellv1(embedding_dim, hidden_dim, n_hiddens, 12)
        elif rnn == 'lstmv4':
            self.rnn_cell = durationLSTMCellv4(embedding_dim, hidden_dim, n_hiddens, 12)
        self.dropout_emb = nn.Dropout(dropout_rate_emb)
        self.dropout_layer = nn.Dropout(dropout_rate_layer)
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim*n_hiddens, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, len(vocab))
        ) 
        
        if tie_weights:
            # assert embedding_dim == hidden_dim*n_hiddens, 'cannot tie, check dims'
            self.fc[2].weight = self.embedding.weight  

    def forward(self, src, hidden, fix_duration):
        # src  (batch_size, seq_len)
        # hidden v1 (n_hiddens, batch_size, d_h) v2 (batch_size, n_hiddens*d_h)
        # fix_duration  (batch_size, seq_len)
        batch_size = src.size(0)

        embedding = self.dropout_emb(self.embedding(src))

        output = []
        for i in range(embedding.size(1)):
            hidden = self.rnn_cell(embedding[:,i:i+1], hidden, fix_duration[:, i])
            if self.rnn == 'v1' or self.rnn == 'v4':
                output.append(hidden.transpose(0,1).contiguous().view(batch_size, -1).unsqueeze(1))
            elif self.rnn == 'v2':
                output.append(hidden.unsqueeze(1))
            elif self.rnn == 'lstmv1' or self.rnn == 'lstmv4':
                output.append(hidden[0].transpose(0,1).contiguous().view(batch_size, -1).unsqueeze(1))
        output = torch.cat(output, dim=1)   # batch_size, seq_len, d_h*n_hidden
             
        output = self.dropout_layer(output) 
        prediction = self.fc(output)
        return prediction, hidden

    def init_hidden(self, batch_size, device):
        return self.rnn_cell.init_hidden(batch_size, device)

    def detach_hidden(self, hidden):
        if self.rnn in ['v1', 'v2', 'v4']:
            return hidden.detach()
        elif self.rnn in ['lstmv1', 'lstmv4']:
            return (hidden[0].detach(), hidden[1].detach())


class durationCellv3(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, rnn):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        assert input_size == hidden_size
        self.rnn = rnn
        if rnn == 'rnn':
            self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        elif rnn == 'lstm':
            self.rnn_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_x, input_h, mask):
        # input_x (batch_size, d_i)
        # input_h (batch_size, d_h)  or  (.,.)
        # mask (batch_size,)    True: compute, False: pass
        mask = mask.float().unsqueeze(1)
        if self.rnn == 'rnn':
            h = self.rnn_cell(input_x, input_h)
            output_x = input_x * (1 - mask) + self.dropout(h) * mask
            output_h = input_h * (1 - mask) + h * mask
        elif self.rnn == 'lstm':
            in_h, in_c = input_h
            h, c = self.rnn_cell(input_x, (in_h, in_c))
            output_x = input_x * (1 - mask) + self.dropout(h) * mask
            out_h = in_h * (1 - mask) + h * mask
            out_c = in_c * (1 - mask) + c * mask
            output_h = (out_h, out_c)
        return output_x, output_h

class durationLMMultiLayer(nn.Module):
    def __init__(self, rnn, vocab, embedding_dim, hidden_dim, num_layers, dropout_rate_emb, dropout_rate_layer,
                tie_weights, predict_on_hidden=True, pretrained_embedding=True, max_duration=12):        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.rescale_rate = max_duration // num_layers
        self.rnn = rnn
        self.predict_on_hidden = predict_on_hidden

        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        if pretrained_embedding:
            word2emb = read_embedding()
            for i in range(len(vocab)):
                word = vocab.lookup_token(i).lower()
                if word in word2emb:
                    self.embedding.weight.data[i] = word2emb[word]
        if rnn == 'rnn':
            self.rnn_list = nn.ModuleList([nn.RNN(embedding_dim, hidden_dim, batch_first=True),])
        elif rnn == 'lstm':
            self.rnn_list = nn.ModuleList([nn.LSTM(embedding_dim, hidden_dim, batch_first=True),])

        for i in range(num_layers-1):
            self.rnn_list.append(durationCellv3(hidden_dim, hidden_dim, dropout_rate_layer, rnn))
        self.dropout_emb = nn.Dropout(dropout_rate_emb)
        self.dropout_layer = nn.Dropout(dropout_rate_layer)
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim*num_layers if predict_on_hidden else hidden_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, len(vocab))
        ) 
        
        if tie_weights:
            # assert embedding_dim == hidden_dim*n_hiddens, 'cannot tie, check dims'
            self.fc[2].weight = self.embedding.weight  

    def forward(self, src, hidden, fix_duration):
        # src  (batch_size, seq_len)
        # hidden [hi,...] (1, batch_size, d_h)
        # fix_duration  (batch_size, seq_len)
        seq_len = src.size(1)
        input_x = self.dropout_emb(self.embedding(src))

        last_hidden = []
        output_hidden = []
        fix_duration = torch.div(fix_duration, self.rescale_rate, rounding_mode='floor')
        for i in range(self.num_layers):
            if i == 0:
                input_x, h = self.rnn_list[i](input_x, hidden[0])
                output_hidden.append(input_x)
                input_x = self.dropout_layer(input_x)
            else:
                output_x = []
                output_h = []
                h = hidden[i]
                for j in range(seq_len):
                    mask = fix_duration[:,j] >= i
                    x, h = self.rnn_list[i](input_x[:,j], h, mask)
                    output_x.append(x.unsqueeze(1))
                    if self.rnn == 'rnn':
                        output_h.append(h.unsqueeze(1))
                    elif self.rnn == 'lstm':
                        output_h.append(h[0].unsqueeze(1))
                input_x = torch.cat(output_x, dim=1)
                output_hidden.append(torch.cat(output_h, dim=1))
            last_hidden.append(h)
        output_hidden = torch.cat(output_hidden, dim=2)
            
        if self.predict_on_hidden:
            prediction = self.fc(self.dropout_layer(output_hidden))
        else:
            prediction = self.fc(self.dropout_layer(input_x))
        return prediction, last_hidden

    def init_hidden(self, batch_size, device):
        if self.rnn == 'rnn':
            return [torch.zeros(1, batch_size, self.hidden_dim).to(device)] + \
                [torch.zeros(batch_size, self.hidden_dim).to(device) for i in range(self.num_layers-1)]
        elif self.rnn == 'lstm':
            return [(torch.zeros(1, batch_size, self.hidden_dim).to(device), \
                torch.zeros(1, batch_size, self.hidden_dim).to(device))] + \
                    [(torch.zeros(batch_size, self.hidden_dim).to(device), \
                torch.zeros(batch_size, self.hidden_dim).to(device)) for i in range(self.num_layers-1)]

    def detach_hidden(self, hidden):
        if self.rnn == 'rnn':
            return [h.detach() for h in hidden]
        elif self.rnn == 'lstm':
            return [(h[0].detach(), h[1].detach()) for h in hidden]

        

class baselineLM(nn.Module):
    def __init__(self, rnn, vocab, embedding_dim, hidden_dim, num_layers, dropout_rate_emb, dropout_rate_layer, 
                tie_weights, pretrained_embedding=True):
                
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        if pretrained_embedding:
            word2emb = read_embedding()
            for i in range(len(vocab)):
                word = vocab.lookup_token(i).lower()
                if word in word2emb:
                    self.embedding.weight.data[i] = word2emb[word]  
        if rnn == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate_layer)  # RNN Test Perplexity: 217.352
        elif rnn == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate_layer) # Test Perplexity: 160.836
        self.dropout_emb = nn.Dropout(dropout_rate_emb)
        self.dropout_layer = nn.Dropout(dropout_rate_layer)
        # self.fc = nn.Linear(hidden_dim, vocab_size) # Test Perplexity: 178.961
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, len(vocab))
        )   # Test Perplexity: 183.213
        
        if tie_weights:
            # assert embedding_dim == hidden_dim, 'cannot tie, check dims'
            self.fc[2].weight = self.embedding.weight  

    def forward(self, src, hidden):
        embedding = self.dropout_emb(self.embedding(src))
        output, hidden = self.rnn(embedding, hidden)          
        output = self.dropout_layer(output) 
        prediction = self.fc(output)
        return prediction, hidden

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        if type(self.rnn) == nn.LSTM:
            cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return hidden, cell
        else:
            return hidden

    def detach_hidden(self, hidden):
        if type(self.rnn) == nn.LSTM:
            hidden, cell = hidden
            return hidden.detach(), cell.detach()
        else:
            return hidden.detach()

if __name__ == '__main__':
    # gradient testing
    torch.autograd.set_detect_anomaly(True)
    vocab = [0] * 1000
    model = durationLM('lstmv1', vocab, 100, 25, 4, None, 0.5, True, False)
    model.eval()
    batch_size = 32
    hidden = model.init_hidden(batch_size, 'cpu')
    hidden[0].requires_grad_()
    src = torch.arange(batch_size*10).long().view(batch_size, -1)
    fix = torch.randint(12, src.size())
    prediction, _ =  model(src, hidden, fix)

    loss = prediction.sum()
    loss.backward()

    print(hidden[0].grad[2,10,0])

    hidden2 = model.init_hidden(batch_size, 'cpu')
    delta = 1e-3
    hidden2[0][2,10,0] += delta
    prediction, _ =  model(src, hidden2, fix)

    loss2 = prediction.sum()

    print((loss2.item() - loss.item())/delta)
    
    # model = durationLMMultiLayer('lstm', vocab, 100, 100, 4, 0.5, 0.25, True, pretrained_embedding=False)
    # model.eval()

    # batch_size = 32
    # hidden = model.init_hidden(batch_size, 'cpu')
    # hidden[1][0].requires_grad_()
    # src = torch.arange(batch_size*10).long().view(batch_size, -1)
    # fix = torch.randint(12, src.size())
    # prediction, _ =  model(src, hidden, fix)

    # loss = prediction.sum()
    # loss.backward()

    # print(hidden[1][0].grad[2,10])

    # hidden2 = model.init_hidden(batch_size, 'cpu')
    # delta = 1e-3
    # hidden2[1][0][2,10] += delta
    # prediction, _ =  model(src, hidden2, fix)

    # loss2 = prediction.sum()

    # print((loss2.item() - loss.item())/delta)