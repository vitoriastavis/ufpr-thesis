from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import torch
import numpy as np
import pandas as pd

def grover(sequence, tokenizer, model, pooling):
    
  inputs = tokenizer(sequence, return_tensors = 'pt')["input_ids"]
  hidden_states = model(inputs)[0] # [1, sequence_length, 768]

#   print(hidden_states.shape)

  # embedding with mean pooling
  if pooling == 'mean':
    embedding = torch.mean(hidden_states[0], dim=0)

  # embedding with max pooling
  else:
    embedding = torch.max(hidden_states[0], dim=0)[0]

  return embedding.detach().numpy()

# Função para ler o CSV e gerar os encodings
def process_sequences(x_train, x_eval, pooling):

    if pooling != 'mean' and pooling != 'max':
        raise TypeError(f"pooling must be 'mean' or 'max'")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER",trust_remote_code=True)
    model = AutoModel.from_pretrained("PoetschLab/GROVER",trust_remote_code=True)
    
    # Apply encoding to train and eval
    encoded_train = [grover(seq, tokenizer, model, pooling) for seq in x_train]    
    encoded_eval = [grover(seq, tokenizer, model, pooling) for seq in x_eval]    

    return encoded_train, encoded_eval

