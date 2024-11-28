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
def process_sequences(x_train, x_eval):

    # Aplica o one-hot encoding nas sequências
    encoded_train = np.array([one_hot(seq) for seq in x_train])
    encoded_eval = np.array([one_hot(seq) for seq in x_eval])

    return encoded_train, encoded_eval
