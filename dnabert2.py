import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from transformers.models.bert.configuration_bert import BertConfig

def dnabert2(sequence, tokenizer, model, pooling):

  inputs = tokenizer(sequence, return_tensors = 'pt')["input_ids"]
  hidden_states = model(inputs)[0] # [1, sequence_length, 768]

  # embedding with mean pooling
  if pooling == 'mean':
    embedding = torch.mean(hidden_states[0], dim=0)

  # embedding with max pooling
  else:
    embedding = torch.max(hidden_states[0], dim=0)[0]

  return embedding.detach().numpy()

# Função para ler o CSV e gerar os encodings
def process_csv(train_file, eval_file, pooling):

    if pooling != 'mean' and pooling != 'max':
        raise TypeError(f"pooling must be 'mean' or 'max'")
    
    # Read files for classifier training and classifier eval

    df_train = pd.read_csv(train_file)    
    df_eval = pd.read_csv(eval_file) 

    x_train = df_train['sequence']
    y_train = df_train['label']

    x_eval = df_eval['sequence']
    y_eval = df_eval['label'] 

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

    # Apply encoding to train and eval
    encoded_train = [dnabert2(seq, tokenizer, model, pooling) for seq in x_train]    
    encoded_eval = [dnabert2(seq, tokenizer, model, pooling) for seq in x_eval]    

    return encoded_train, y_train, encoded_eval, y_eval
