import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.tokenize import NLTKWordTokenizer 
import random
import os
import pickle


class dictionary():
    def __init__(self):
        self.word_freq={}
        self.id2word={}
        self.word2id={}
    
    def add_word(self,word):
        self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
    def create_mapping(self):
        self.word_freq['[PAD]']=1000001
        self.word_freq['[UNK]']=1000000
        c_unk=0
        dic_items=[]
        for k in self.word_freq.keys():
            if self.word_freq[k]>1 or np.random.uniform()>0.5:
                dic_items.append((k,self.word_freq[k]))
            else:
                c_unk+=1
        ordered_lis=sorted( dic_items, key=lambda x: (-x[1],x[0]))
        assert ordered_lis[0][0]=='[PAD]'
        self.id2word=dict([(i,ordered_lis[i][0]) for i in range(len(ordered_lis))])
        self.word2id=dict([(ordered_lis[i][0],i) for i in range(len(ordered_lis))])
        self.ordered_lis=ordered_lis

        self.pad_id = 0
        self.unk_id = 1
        return c_unk

    def get_id(self,word):
        return self.word2id.get(word, 1)
    
    def get_word(self,idx):
        return self.id2word[idx]
    
    def get_len(self):
        return len(self.id2word)

    def __len__(self):
        return len(self.id2word)

class subset(Dataset):
    def __init__(self, input_tokens, labels, mapping):
        super().__init__()
        self.input_tokens = input_tokens
        self.labels = labels
        self.data_num=len(input_tokens)

        """convert X(token) to X(id)"""
        if mapping != None:
            self.input_ids=[]
            for i in range(len(input_tokens)):
                self.input_ids.append(list(map(lambda x:mapping.get_id(x), input_tokens[i])))

    def __getitem__(self,index):
        return self.input_ids[index], self.labels[index]

    def __len__(self):
        return self.data_num

class fixationPredictionDataset():
    def __init__(self, feature, average=False, test_proportion=0.25, max_len=512, shuffle=True):

        if feature not in ['nFixations', 'GD', 'TRT', 'FFD', 'SFD', 'GPT', 'TRT-noD']:
            raise RuntimeError('select a valid feature')
        self.feature = feature

        path = './FPdatasets/FPdata-%s-%s-test%.2f-%s-pickle'%\
            (feature, 'averaged' if average else '', test_proportion, 'shuffled' if shuffle else '')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                obj = pickle.load(f)
                self.train_set, self.test_set, self.mapping, self.max_len = obj
            return 

        self.max_len = max_len
        """load data"""
        s_time = time.time()
        allData = pd.read_parquet('allData.parquet')
        print('time spend on loading: ', time.time() - s_time)

        if feature in ['GD', 'GPT', 'TRT-noD']:
            allData.drop(index=allData.loc[allData['Corpus']=='Dundee'].index, inplace=True)
        # some checks
        # allData.hist(column=feature, bins=12, by='Corpus', range=(0, 800), figsize=(8,10))
        # plt.show()

        print(allData.groupby('Corpus')[feature].mean())    # do norm on corpus level or not?
        allData[feature] = allData[feature] / allData.groupby('Corpus')[feature].transform('mean') * 100
        print(allData.groupby('Corpus')[feature].mean())

        """average over subjects"""
        if average:
            grouped = allData.groupby(['Corpus','Sent_ID','Word_ID'], as_index=False, sort=False)
            df1 = grouped['Word'].agg(lambda x: pd.Series.mode(x)[0])
            cols = ['Subj_ID',  'nFixations', 'GD',  'TRT',  'FFD',  'SFD',  'GPT']
            df2 = grouped[cols].mean()
            allData = pd.concat([df1, df2[cols]], axis=1)
            print(allData)
            # print(grouped.loc[ (grouped['Subj_ID']!=12) & (grouped['Subj_ID']!=18) & (grouped['Subj_ID']!=10) & (grouped['Subj_ID']!=14)  ])
            
            # print(allData.loc[(allData['Corpus']=='Dundee') & (allData['Subj_ID']==) & (allData['Sent_ID']=='119')].to_string())

        """map the feature values to 0,1,2...11"""
        if average:
            quant = allData[feature].quantile(np.linspace(0,1,13)[1:-1])  # 12 intervals
            # quant = [0] + quant.tolist() + [800]
            # allData.hist(column=feature, bins=quant)
            # plt.show()
            # exit()
            quant = quant.tolist()
        else:
            nonz_values = allData[feature].loc[allData[feature]>0]
            quant = nonz_values.quantile(np.linspace(0,1,12)[1:-1])    # 11 intervals
            # quant = [0, 20] + quant.tolist() + [800]
            # allData.hist(column=feature, bins=quant)
            # plt.show()
            # exit()
            quant = quant.tolist()
            quant.insert(0, 1)
        print(quant)
        
        allData['target'] = 0
        for i, quantile in enumerate(quant):
            allData.loc[allData[feature] >= quantile, 'target'] = i + 1
        print(allData.head(10))

        """convert to X(token) and Y"""
        s_time = time.time()
        tokenizer = NLTKWordTokenizer()
        X = []
        Y = []
        Sent_ID = ''
        data_dict = allData.to_dict('records')
        for row in tqdm(data_dict):
            if row['Sent_ID'] != Sent_ID:
                Sent_ID = row['Sent_ID']
                if len(X) > 0 and len(X[-1]) > self.max_len:
                    X[-1] = X[-1][:self.max_len]
                    Y[-1] = Y[-1][:self.max_len]
                    print("truncate")
                X.append([])
                Y.append([])      
            # consider some cases: 'word  "word  word,   word.  don't  I'm  word-word
            # word = re.sub(r'\d','0',row['Word'].lower())
            word = row['Word']
            word = re.sub(r'-', ' - ', word)
            tokens = tokenizer.tokenize(word)
            for token in tokens:
                X[-1].append(token)
                if re.search(r'\w',token):
                    Y[-1].append(row['target'])
                else:
                    Y[-1].append(1)
        print('convert to X and Y: ', time.time()-s_time)
        print('number of sentences: ', len(X))

        """split data into train & test"""
        test_number = int(len(X) * test_proportion)
        if shuffle:
            indices = list(range(len(X)))
            random.shuffle(indices)
            test_indices = indices[:test_number]
            train_indices = indices[test_number:]
            X_train = list(map(lambda i: X[i], train_indices))
            Y_train = list(map(lambda i: Y[i], train_indices))

            X_test = list(map(lambda i: X[i], test_indices))
            Y_test = list(map(lambda i: Y[i], test_indices))
        else:
            X_train = X[test_number:]
            Y_train = Y[test_number:]

            X_test = X[:test_number]
            Y_test = Y[:test_number]

        """create a mapping between token and id"""
        mapping=dictionary()
        for sentence in X_train:
            for word in sentence:
                mapping.add_word(word)
        mapping.create_mapping()

        
        self.train_set = subset(X_train, Y_train, mapping)
        self.test_set = subset(X_test, Y_test, mapping)

        self.mapping=mapping

        with open(path, 'wb') as f:
            pickle.dump((self.train_set, self.test_set, self.mapping, self.max_len), f)
        print('data dumped at %s'%path)


class fixationPredictionDatasetVar():
    def __init__(self, feature, test_proportion=0.25, max_len=512, shuffle=True):

        if re.search('nFixations|GD|TRT|FFD|SFD|GPT', feature) is None:
            raise RuntimeError('select a valid feature')
        self.feature = feature

        path = './FPdatasets/FPdata-var-%s-test%.2f-%s-pickle'%\
            (feature, test_proportion, 'shuffled' if shuffle else '')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                obj = pickle.load(f)
                self.train_set, self.test_set, self.mapping, self.max_len = obj
            return 

        self.max_len = max_len
        """load data"""
        s_time = time.time()
        allData = pd.read_parquet('allData.parquet')
        print('time spend on loading: ', time.time() - s_time)

        if re.search('GD|GPT', feature):
            allData.drop(index=allData.loc[allData['Corpus']=='Dundee'].index, inplace=True)
            print('drop Dundee')

        if re.search('-noD', feature):
            allData.drop(index=allData.loc[allData['Corpus']=='Dundee'].index, inplace=True)
            print('drop Dundee')
            feature = re.sub('-noD', '', feature)

        if re.search('-noG', feature):
            allData.drop(index=allData.loc[allData['Corpus']=='GECO'].index, inplace=True)
            print('drop GECO')
            feature = re.sub('-noG', '', feature) 

        # some checks
        # allData.hist(column=feature, bins=12, by='Corpus', range=(0, 800), figsize=(8,10))
        # plt.show()

        print(allData.groupby('Corpus')[feature].mean())    # do norm on corpus level or not?
        allData[feature] = allData[feature] / allData.groupby('Corpus')[feature].transform('mean') * 100
        print(allData.groupby('Corpus')[feature].mean())

        """average over subjects"""
        grouped = allData.groupby(['Corpus','Sent_ID','Word_ID'], as_index=False, sort=False)
        grouped = grouped.filter(lambda x: len(x) > 3 )
        grouped = grouped.groupby(['Corpus','Sent_ID','Word_ID'], as_index=False, sort=False)
        allData = grouped['Word'].agg(lambda x: pd.Series.mode(x)[0])
        
        allData['target'] = grouped[feature].mean()[feature]
        allData['variance'] = grouped[feature].var()[feature]

        """convert to X(token) and Y"""
        s_time = time.time()
        tokenizer = NLTKWordTokenizer()
        X = []
        Y = []
        Sent_ID = ''
        data_dict = allData.to_dict('records')
        for row in tqdm(data_dict):
            if row['Sent_ID'] != Sent_ID:
                Sent_ID = row['Sent_ID']
                if len(X) > 0 and len(X[-1]) > self.max_len:
                    X[-1] = X[-1][:self.max_len]
                    Y[-1] = Y[-1][:self.max_len]
                    print("truncate")
                X.append([])
                Y.append([])      
            # consider some cases: 'word  "word  word,   word.  don't  I'm  word-word
            # word = re.sub(r'\d','0',row['Word'].lower())
            word = row['Word']
            word = re.sub(r'-', ' - ', word)
            tokens = tokenizer.tokenize(word)
            for token in tokens:
                X[-1].append(token)
                if re.search(r'\w',token):
                    Y[-1].append((row['target'], row['variance']))
                else:
                    Y[-1].append((0, float('inf')))
        print('convert to X and Y: ', time.time()-s_time)
        print('number of sentences: ', len(X))

        """split data into train & test"""
        test_number = int(len(X) * test_proportion)
        if shuffle:
            indices = list(range(len(X)))
            random.shuffle(indices)
            test_indices = indices[:test_number]
            train_indices = indices[test_number:]
            X_train = list(map(lambda i: X[i], train_indices))
            Y_train = list(map(lambda i: Y[i], train_indices))

            X_test = list(map(lambda i: X[i], test_indices))
            Y_test = list(map(lambda i: Y[i], test_indices))
        else:
            X_train = X[test_number:]
            Y_train = Y[test_number:]

            X_test = X[:test_number]
            Y_test = Y[:test_number]

        """create a mapping between token and id"""
        
        self.train_set = subset(X_train, Y_train, None)
        self.test_set = subset(X_test, Y_test, None)


        with open(path, 'wb') as f:
            pickle.dump((self.train_set, self.test_set, None, self.max_len), f)
        print('data dumped at %s'%path)

def collate_batch(batch):
    #input is a list of tuples
    word_num=list(map(lambda x:len(x[0]),batch))
    max_word_num=max(word_num)
    input_ids=list(map(lambda x:x[0]+[0]*(max_word_num-len(x[0])), batch))
    labels=list(map(lambda x:x[1]+[0]*(max_word_num-len(x[1])), batch))

    input_ids=torch.LongTensor(input_ids)
    labels=torch.LongTensor(labels)

    return input_ids, labels

        





if __name__ == '__main__':
    FPD = fixationPredictionDataset('TRT',average=True)
    for i in range(10):
        index = random.randint(0, len(FPD.train_set)-1)
        print(list(zip(FPD.train_set.input_tokens[index], FPD.train_set.labels[index])))