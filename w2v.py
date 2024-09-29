import nltk
import urllib
import bs4 as bs
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

def w2v(model, word):   

    if word in model.wv:
        # Return the word vector for the specified word
        return model.wv[word]
    else:
        return None

# Função para ler o CSV e gerar os encodings
def process_csv(train_file, encode_file):
    # Lê o CSV ignorando a primeira linha
    df_train = pd.read_csv(train_file)    
    df_encode = pd.read_csv(encode_file)

    train = df_train['sequence']
    
    vector_size = 250
    window = 20
    min_count = 1
    epochs = 150

    model = Word2Vec(train, vector_size=vector_size,
                    window=window, min_count=min_count, sg=1)
    model.train(train, total_examples=len(train), epochs=epochs)
    
    # Aplica o word2vec nas sequencias 
    encoded_sequences = np.array([w2v(model, seq) for seq in df_encode['sequence']])
            
    labels = df_encode['label']  
    
    return encoded_sequences, labels


