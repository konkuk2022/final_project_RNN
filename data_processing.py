import pytorch_lightning as pl
import torch.nn as nn
from transformers import ElectraModel, AutoTokenizer
import torch

import kss

import pandas as pd

def one_hot_encode(data, n_label=44):
    data = list(map(int,data.split(',')))
    one_hot = [0] * n_label
    label_idx = data
    for idx in label_idx:
        one_hot[idx] = 1
    return torch.LongTensor(one_hot)

def data_preprocessing(df,electra_model):
    # text split // 24분 소요
    X = df['text'][:]
    pr_x = []
    for x in X:
        split_x = []
        sp_x = kss.split_sentences(x)
        for sent in sp_x:
            split_x.append(electra_model(sent)[0].tolist())
        pr_x.append(split_x)

    # label one hot encoding and to tensor
    Y = df['emotion'][:]
    pr_y = []
    for y in Y:
        pr_y.append(one_hot_encode(y))
    
    return pr_x, pr_y


device = "mps" if torch.backends.mps.is_available() else "cpu"

class KOTEtagger(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.electra = ElectraModel.from_pretrained("beomi/KcELECTRA-base").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base", TOKENIZERS_PARALLELISM=True)
        self.classifier = nn.Linear(self.electra.config.hidden_size, 44).to(device)
        
    def forward(self, text:str):
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=512,
          return_token_type_ids=False,
          padding="max_length",
          return_attention_mask=True,
          return_tensors='pt',
        ).to(device)
        output = self.electra(encoding["input_ids"], attention_mask=encoding["attention_mask"])
        output = output.last_hidden_state[:,0,:]
        output = self.classifier(output)
        # output = torch.sigmoid(output)
        torch.cuda.empty_cache()
        
        return output
    
electra_model_path= "./saved_model/kote_pytorch_lightning.bin"
data_type = 'train' # train / test /  
data_path = f'./data/{data_type}.tsv'

df = pd.read_csv(data_path,delimiter='\t',names=['num','text','emotion'],header=None)

trained_model = KOTEtagger()
trained_model.load_state_dict(torch.load(electra_model_path))

x_data,y_data = data_preprocessing(df,trained_model)
data_xy = pd.DataFrame(list(zip(x_data,y_data)), columns = ['emotion','label'])
data_xy.to_pickle(f"./data/{data_type}_data.pkl")


data_type = 'test' # train / test / val 
data_path = f'./data/{data_type}.tsv'

df2 = pd.read_csv(data_path,delimiter='\t',names=['num','text','emotion'],header=None)

x_data2,y_data2 = data_preprocessing(df2,trained_model)
data_xy2 = pd.DataFrame(list(zip(x_data2,y_data2)), columns = ['emotion','label'])
data_xy2.to_pickle(f"./data/{data_type}_data.pkl")

data_type = 'val' # train / test /
data_path = f'./data/{data_type}.tsv'

df3 = pd.read_csv(data_path,delimiter='\t',names=['num','text','emotion'],header=None)

x_data3,y_data3 = data_preprocessing(df3,trained_model)
data_xy3 = pd.DataFrame(list(zip(x_data3,y_data3)), columns = ['emotion','label'])
data_xy3.to_pickle(f"./data/{data_type}_data.pkl")
