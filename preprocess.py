from scripts_ZuCo import utils_ZuCo
from scripts_ZuCo2 import data_loading_helpers as dh
from tqdm import tqdm
import os
import numpy as np
import h5py
import pandas as pd

datatransform_t1 = utils_ZuCo.DataTransformer('task1', level='word', scaling='raw')
subjects_t1 = []
for i in tqdm(range(12)):
    subjects_t1.append( datatransform_t1(i) )

for i in range(len(subjects_t1)):
    subjects_t1[i].drop(columns=['meanPupilSize', 'WordLen'], inplace=True)
    subjects_t1[i].insert(0, 'Subj_ID', i)

subjects_t1 = pd.concat(subjects_t1, ignore_index=True)
subjects_t1.insert(0, 'Corpus', 'Z1T1')
print(subjects_t1)

# print( subjects_t1[['nFixations','FFD','SFD']].apply(lambda x: x[1]==x[2] if x[0]==1 else 0==x[2], axis=1).all() ) # True
# print( subjects_t1.iloc[:,5:].apply(lambda x: (x==0).all() if x[0]==0 else True, axis=1).all() ) # True
'''
  Sent_ID Word_ID      Word nFixations meanPupilSize   GD  TRT  FFD  SFD  GPT WordLen
0    0_NR       0  presents          4    934.250000  119  669  119    0  119       8
1    0_NR       1         a          3    945.666667   73  295   73    0  845       1
2    0_NR       2      good          0      0.000000    0    0    0    0    0       4
3    0_NR       3      case          1   1013.000000  106  106  106  106  106       4
4    0_NR       4     while          3    961.000000  333  333  145    0  333       5
5    0_NR       5   failing          0      0.000000    0    0    0    0    0       7
6    0_NR       6        to          1    921.000000   79   79   79   79   79       2
7    0_NR       7   provide          1    906.000000   95   95   95   95   95       7
8    0_NR       8         a          1    885.000000   82   82   82   82   82       1
9    0_NR       9    reason          2    839.500000  155  155   76    0  155       6
'''

datatransform_t2 = utils_ZuCo.DataTransformer('task2', level='word', scaling='raw')
subjects_t2 = []
for i in tqdm(range(12)):
    subjects_t2.append( datatransform_t2(i) )

for i in range(len(subjects_t2)):
    subjects_t2[i].drop(columns=['meanPupilSize', 'WordLen'], inplace=True)
    subjects_t2[i].insert(0, 'Subj_ID', i)

subjects_t2 = pd.concat(subjects_t2, ignore_index=True)
subjects_t2.insert(0, 'Corpus', 'Z1T2')
print(subjects_t2)



rootdir = "ZuCo2-task1/"
subj_idx = 0
task_dfs = []
for file in tqdm(os.listdir(rootdir)):
    if file.endswith("NR.mat"):
        print(file)

        file_name = rootdir + file
        subject = file_name.split("ts")[1].split("_")[0]

        # exclude YMH due to incomplete data because of dyslexia
        if subject != 'YMH':

            f = h5py.File(file_name)
            sentence_data = f['sentenceData']
            # contentData = sentence_data['content']
            wordData = sentence_data['word']
            subject_dfs = []

            for idx in range(len(wordData)):  # number of sentences:
                # obj_reference_content = contentData[idx][0]
                # sent = dh.load_matlab_string(f[obj_reference_content])
                # print(sent)

                # get word level data
                word_data = dh.extract_word_level_data(f, f[wordData[idx][0]])

                # number of tokens
                # print(len(word_data))
                # print(word_data[0].keys())
                "dict_keys(['RAW_EEG', 'RAW_ET', 'FFD', 'GD', 'GPT', 'TRT', 'SFD', 'nFix', 'ALPHA_EEG', 'BETA_EEG', 'GAMMA_EEG', 'THETA_EEG', 'word_idx', 'content'])"

                fields = ['Sent_ID', 'Word_ID', 'Word', 'nFixations', 'GD', 'TRT', 'FFD', 'SFD', 'GPT']
                df = pd.DataFrame(index=range(len(word_data)), columns=fields)

                for widx in range(len(word_data)):
                    df.iloc[widx, 0] = str(idx) + '_NR'
                    df.iloc[widx, 1] = widx
                    df.iloc[widx, 2] = word_data[widx]['content']
                    features = [word_data[widx]['nFix'], word_data[widx]['GD'], word_data[widx]['TRT'],\
                                        word_data[widx]['FFD'], word_data[widx]['SFD'], word_data[widx]['GPT']]
                    df.iloc[widx, 3:] = list(map(lambda x: int(x) if x is not None else 0, features))

                    assert widx == word_data[widx]['word_idx']
            
                df.fillna(0, inplace=True)
                subject_dfs.append(df)
            
            subject_dfs = pd.concat(subject_dfs, ignore_index=True)
            subject_dfs.insert(0, 'Subj_ID', subj_idx)
            subj_idx += 1
            task_dfs.append(subject_dfs)

task_dfs = pd.concat(task_dfs, ignore_index=True)
task_dfs.insert(0, 'Corpus', 'Z2T1')
print(task_dfs)


df = pd.read_excel('./GeCo/MonolingualReadingData.xlsx')

subj_ID = df.PP_NR.unique()
mapping = dict([ (subj, i) for i, subj in enumerate(subj_ID) ])

# "Corpus Subj_ID Sent_ID Word_ID  Word nFixations   GD  TRT  FFD  SFD  GPT"

geco_df = pd.DataFrame({'Corpus':'GECO', 'Subj_ID': df['PP_NR'].apply(lambda x: mapping[x]), 'Sent_ID':df[['PART','TRIAL']].apply(lambda x: str(x[0])+'-'+str(x[1]), axis=1), \
                'Word_ID': df['WORD_ID_WITHIN_TRIAL']-1, 'Word':df['WORD'].astype('str'), 'nFixations':df['WORD_FIXATION_COUNT'],\
                'GD': df['WORD_GAZE_DURATION'], 'TRT':df['WORD_TOTAL_READING_TIME'], 'FFD':df['WORD_FIRST_FIXATION_DURATION'], \
                'SFD': df[['WORD_FIRST_FIXATION_DURATION','WORD_FIXATION_COUNT']].apply(lambda x: x[0] if x[1] == 1 else 0, axis=1), \
                'GPT':df['WORD_GO_PAST_TIME'] } )

def make_zero(line):
    result = line.copy()
    if line[5] == 0:    # nFixations is 0
        result[6:] = 0
    return result
geco_df = geco_df.apply(make_zero, axis=1)

print(geco_df)

df = pd.read_csv('./Dundee/EN_dundee.csv', sep='\t')
print('dundee df loaded')

# FFD: First_fix_dur
# TRT: Tot_fix_dur
# nFixations: nFix
# GD: Tot_fix_dur - Tot_regres_to_dur  guess, not sure, so do not use it
# GPT: GD + Tot_regres_from_dur   guess, not sure, so do not use it

subj_ID = df.Participant.unique()
mapping = dict([ (subj, i) for i, subj in enumerate(subj_ID) ])

dundee_df = pd.DataFrame({'Corpus':'Dundee', 'Subj_ID': df['Participant'].apply(lambda x: mapping[x]), \
                'Sent_ID':df[['Itemno','SentenceID']].astype('int').apply(lambda x: '%d-%d'%(x[0], x[1]), axis=1), \
                'Word_ID': df['ID']-1, 'Word':df['WORD'], 'nFixations':df['nFix'],\
                'GD': 0, 'TRT':df['Tot_fix_dur'], 'FFD':df['First_fix_dur'], \
                'SFD': df[['First_fix_dur','nFix']].apply(lambda x: x[0] if x[1] == 1 else 0, axis=1), \
                'GPT': 0, } )
print('dundee df complete phase 1')
last_item = ''
idx = 0
Sent_ID = ''
Word_ID = 0
while idx < len(dundee_df):
    word = dundee_df.iloc[idx,4]
    if word == last_item:
        dundee_df.drop(index=[dundee_df.index[idx]],inplace=True)
        continue

    last_item = word

    if Sent_ID != dundee_df.iloc[idx,2]:
        Sent_ID = dundee_df.iloc[idx,2]
        Word_ID = 0
    dundee_df.iloc[idx,3] = Word_ID
    Word_ID += 1

    idx += 1

print('dundee df complete phase 2')
dundee_df = dundee_df.apply(make_zero, axis=1)

dundee_df = dundee_df.astype({'Word_ID': 'int64', 'nFixations':'int64', 'TRT':'int64', 'FFD':'int64', 'SFD':'int64'})

print(dundee_df)

# for item in [subjects_t1, subjects_t2, task_dfs, geco_df, dundee_df]:
#     print(item.columns)

whole_df = pd.concat([subjects_t1, subjects_t2, task_dfs, geco_df, dundee_df], ignore_index=True)
whole_df.to_parquet('allData.parquet')     # to_csv would be slower  

head_len = 50
sample_df = pd.concat([subjects_t1.head(head_len), subjects_t2.head(head_len), task_dfs.head(head_len), geco_df.head(head_len), dundee_df.head(head_len)], ignore_index=True)
sample_df.to_excel('sampleData.xlsx')

# pd.read_parquet('allData.parquet')

