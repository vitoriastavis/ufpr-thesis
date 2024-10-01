import nltk
import urllib
import bs4 as bs
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict

def w2v(model, word):   

    if word in model.wv:
        # Return the word vector for the specified word
        return model.wv[word]
    else:
        return None
    
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def bpe(corpus):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
   

    word_freqs = defaultdict(int)

    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    alphabet = ['A', 'C', 'T', 'G']
    vocab = ["<|endoftext|>"] + alphabet.copy()

    # for word in word_freqs.keys():
    #     for letter in word:
    #         if letter not in alphabet:
    #             alphabet.append(letter)
    # alphabet.sort()

    splits = {word: [c for c in word] for word in word_freqs.keys()}



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


