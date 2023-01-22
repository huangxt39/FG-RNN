import re
from tqdm import tqdm
import torch

def add_fix_duration(dataset, vocab, FPmodel, device):
    data = []                                        
    for example in dataset.input_tokens:
        token_ids = [vocab.get_id( re.sub('[=@]', '-', re.sub(r'\d','0',token.lower())) ) for token in example]
        data.append(token_ids)
    
    # predict in batch
    pred_fix = []
    
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
    assert len(data) == len(pred_fix) == len(dataset)

    for i in range(len(dataset)):
        dataset.fix_duration[i] = pred_fix[i][:len(dataset.input_tokens[i])]
            
    return dataset