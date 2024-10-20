import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Função para realizar one-hot encoding de uma sequência de DNA
def one_hot(sequence):
    mapping = {'A': [1, 0, 0, 0],
               'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'T': [0, 0, 0, 1]}
    
    # Transforma cada base da sequência no vetor correspondente
    return np.array([mapping[base] for base in sequence])

# Função para ler o CSV e gerar os encodings
def process_csv(train_path, eval_path):
    # Lê o CSV ignorando a primeira linha
    df_train = pd.read_csv(train_path)    
    df_eval = pd.read_csv(eval_path)    

    # Aplica o one-hot encoding nas sequências
    encoded_train = np.array([one_hot(seq) for seq in df_train['sequence']])
    encoded_eval = np.array([one_hot(seq) for seq in df_eval['sequence']])

    y_train = df_train['label']
    y_eval = df_eval['label']
    
    return encoded_train, y_train, encoded_eval, y_eval
