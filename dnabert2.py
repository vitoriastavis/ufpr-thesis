import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

def dnabert2(sequence, tokenizer, model, pooling):

  inputs = tokenizer(sequence, return_tensors = 'pt')["input_ids"]
  hidden_states = model(inputs)[0] 

  # embedding with mean pooling
  if pooling == 'mean':
    embedding = torch.mean(hidden_states[0], dim=0)

  # embedding with max pooling
  else:
    embedding = torch.max(hidden_states[0], dim=0)[0]

  return embedding.detach().numpy()

# Receive x_train and x_eval and return their embeddings
# according to the model type (pretrained or finetuned)
def process_sequences(x_train, x_eval, pooling, model_type):

    if pooling != 'mean' and pooling != 'max':
        raise TypeError(f"pooling must be 'mean' or 'max'")

    if model_type == 'pretrained':
       model_path = 'zhihan1996/DNABERT-2-117M'
    elif model_type == 'finetuned-cancer':
       model_path = 'UKaizokuO/DNABERT-2-117M-finetuned-cancer'
    else:
       raise TypeError(f"model_type must be 'pretrained' or 'finetuned-cancer'")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, config=config)

    # Apply encoding to train and eval
    encoded_train = [dnabert2(seq, tokenizer, model, pooling) for seq in x_train]    
    encoded_eval = [dnabert2(seq, tokenizer, model, pooling) for seq in x_eval]    

    return encoded_train, encoded_eval
