from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import torch
from transformers.models.bert.configuration_bert import BertConfig

def grover(sequence, tokenizer, model, pooling):
    
  inputs = tokenizer(sequence, return_tensors = 'pt')["input_ids"]
  hidden_states = model(inputs)[0] 

  # embedding with mean pooling
  if pooling == 'mean':
    embedding = torch.mean(hidden_states[0], dim=0)

  # embedding with max pooling
  else:
    embedding = torch.max(hidden_states[0], dim=0)[0]

  return embedding.detach().numpy()

# Função para ler o CSV e gerar os encodings
def process_sequences(x_train, x_eval, pooling, model_type):

    if pooling != 'mean' and pooling != 'max':
        raise TypeError(f"pooling must be 'mean' or 'max'")
    
    if model_type == 'grover-pretrained':
       model_path = 'PoetschLab/GROVER'
    elif model_type == 'grover-finetuned-cancer':
       model_path = 'UKaizokuO/GROVER-finetuned-cancer'
    else:
       raise TypeError(f"model_type must be 'pretrained' or 'finetuned-cancer'")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    config = BertConfig.from_pretrained(model_path)
    config.alibi_starting_size = 0
    model = AutoModel.from_pretrained(model_path,trust_remote_code=True, config=config)
    
    # Apply encoding to train and eval
    encoded_train = [grover(seq, tokenizer, model, pooling) for seq in x_train]    
    encoded_eval = [grover(seq, tokenizer, model, pooling) for seq in x_eval]    

    return encoded_train, encoded_eval

