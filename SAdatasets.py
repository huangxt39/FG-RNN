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
from FPdataset import dictionary




class subset(Dataset):
    def __init__(self, input_tokens, fix_duration, labels, mapping):
        super().__init__()
        self.input_tokens = input_tokens
        self.fix_duration = fix_duration
        self.labels = labels
        self.data_num=len(input_tokens)

        """convert X(token) to X(id)"""
        if mapping != None:
            self.input_ids=[]
            for i in range(len(input_tokens)):
                self.input_ids.append(list(map(lambda x:mapping.get_id(x), input_tokens[i])))

    def __getitem__(self,index):
        return self.input_ids[index], self.fix_duration[index], self.labels[index]

    def __len__(self):
        return self.data_num

class SentimentAnalysisDataset():
    def __init__(self, test_proportion=0.1, shuffle=True, redo=False):

        path = './Eye-tracking_and_SA-II/SAdata-test%.2f-%s-pickle'%\
            ( test_proportion, 'shuffled' if shuffle else '')
        if os.path.exists(path) and not redo:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
                self.train_set, self.test_set, self.mapping = obj
            return 

        """load data"""
        s_time = time.time()
        allData = pd.read_csv('Eye-tracking_and_SA-II.csv')
        print('time spend on loading: ', time.time() - s_time)
        print(allData)
        print(allData.dtypes)
        """map the feature values to 0,1,2...11"""
        quant = allData['Fixation_Duration'].quantile(np.linspace(0,1,13)[1:-1])  # 12 intervals
        # quant = [0] + quant.tolist() + [800]
        # allData.hist(column=feature, bins=quant)
        # plt.show()
        # exit()
        quant = quant.tolist()
        print(quant)
        
        
        allData['duration'] = 0
        for i, quantile in enumerate(quant):
            allData.loc[allData['Fixation_Duration'] >= quantile, 'duration'] = i + 1
        print(allData.head(10))


        """convert to X(token) and Y"""
        s_time = time.time()
        tokenizer = NLTKWordTokenizer()
        X = []
        D = [] # duration
        Y = []
        Text_ID = ''
        data_dict = allData.to_dict('records')
        for row in tqdm(data_dict):
            if row['Text_ID'] != Text_ID:
                Text_ID = row['Text_ID']
                X.append([])    
                D.append([])  
                Y.append(1 if row['Default_Polarity'] == 1.0 else 0) 
                assert row['Default_Polarity'] == -1.0 or row['Default_Polarity'] == 1.0 
            # consider some cases: 'word  "word  word,   word.  don't  I'm  word-word
            word = row['Word'].lower()
            word = re.sub(r'--', ' -- ', word)
            tokens = tokenizer.tokenize(word)
            for token in tokens:
                X[-1].append(token)
                if re.search(r'\w',token):
                    D[-1].append(row['duration'])
                else:
                    D[-1].append(1)
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
            D_train = list(map(lambda i: D[i], train_indices))
            Y_train = list(map(lambda i: Y[i], train_indices))

            X_test = list(map(lambda i: X[i], test_indices))
            D_test = list(map(lambda i: D[i], test_indices))
            Y_test = list(map(lambda i: Y[i], test_indices))
        else:
            X_train = X[test_number:]
            D_train = D[test_number:]
            Y_train = Y[test_number:]

            X_test = X[:test_number]
            D_test = D[:test_number]
            Y_test = Y[:test_number]

        """create a mapping between token and id"""
        mapping=dictionary()
        for sentence in X_train:
            for word in sentence:
                mapping.add_word(word)
        mapping.create_mapping()

        
        self.train_set = subset(X_train, D_train, Y_train, mapping)
        self.test_set = subset(X_test, D_test, Y_test, mapping)

        self.mapping=mapping

        with open(path, 'wb') as f:
            pickle.dump((self.train_set, self.test_set, self.mapping), f)
        print('data dumped at %s'%path)



def collate_batch(batch):
    #input is a list of tuples
    word_num=list(map(lambda x:len(x[0]),batch))
    max_word_num=max(word_num)
    input_ids=list(map(lambda x:x[0]+[0]*(max_word_num-len(x[0])), batch))
    fix_duration=list(map(lambda x:x[1]+[0]*(max_word_num-len(x[1])), batch))
    input_len = list(map(lambda x: len(x[0]), batch))
    labels = list(map(lambda x:x[2], batch))

    input_ids=torch.LongTensor(input_ids)
    fix_duration=torch.tensor(fix_duration)
    input_len = torch.LongTensor(input_len)
    labels=torch.LongTensor(labels)

    return input_ids, fix_duration, input_len, labels

        





if __name__ == '__main__':
    FPD = SentimentAnalysisDataset(redo=True)
    for i in range(10):
        index = random.randint(0, len(FPD.train_set)-1)
        print(list(zip(FPD.train_set.input_tokens[index], FPD.train_set.fix_duration[index])), FPD.train_set.labels[index])