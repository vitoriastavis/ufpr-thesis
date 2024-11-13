import haiku as hk
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd

def nt(sequence, tokenizer, model):
    # inputs = tokenizer(sequence, return_tensors = 'pt')["input_ids"]
    max_length = tokenizer.model_max_length
    tokens_ids = tokenizer.batch_encode_plus(sequence, return_tensors="pt",
                                            padding="max_length",
                                            max_length = max_length)["input_ids"]

    attention_mask = tokens_ids != tokenizer.pad_token_id

    torch_outs = model(
        tokens_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True
    )
    embedding = torch_outs['hidden_states'][-1]

    # with torch.no_grad():
    #     hidden_states = model(inputs)[0] 

    # # embedding with mean pooling
    # if pooling == 'mean':
    #   embedding = torch.mean(hidden_states[0], dim=0)

    # # embedding with max pooling
    # else:
    #   embedding = torch.max(hidden_states[0], dim=0)[0]

    return embedding.detach().numpy()

# Função para ler o CSV e gerar os encodings
def process_csv(train_file, eval_file):

    # Read files for classifier training and classifier eval
    df_train = pd.read_csv(train_file)    
    df_eval = pd.read_csv(eval_file) 

    x_train = df_train['sequence']
    y_train = df_train['label']

    x_eval = df_eval['sequence']
    y_eval = df_eval['label'] 

    # Import the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
    model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")

    # Apply encoding to train and eval
    encoded_train = [nt(seq, tokenizer, model, pooling) for seq in x_train]    
    encoded_eval = [nt(seq, tokenizer, model, pooling) for seq in x_eval]    

    print(f"Embeddings shape: {encoded_train.shape}")
    
    return encoded_train, y_train, encoded_eval, y_eval

# # Add embed dimension axis
# attention_mask = torch.unsqueeze(attention_mask, dim=-1)

# # Compute mean embeddings per sequence
# mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
# print(f"Mean sequence embeddings: {mean_sequence_embeddings}")
