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
def process_csv(file_path):
    # Lê o CSV ignorando a primeira linha
    df = pd.read_csv(file_path)    
    
    # Aplica o one-hot encoding nas sequências
    encoded_sequences = np.array([one_hot(seq) for seq in df['sequence']])

    # encoded_df = pd.DataFrame(encoded_sequences)
    labels = df['label']
    
    return encoded_sequences, labels
