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

# Aplica o word2vec nas sequencias 
def w2v(token_train, token_encode, vector_size, window, min_count, epochs): 

    
    model = Word2Vec(token_train, vector_size=vector_size,
                    window=window, min_count=min_count, sg=1)
    model.train(token_train, total_examples=len(token_train), epochs=epochs)

    encoded_sequences = []

    for seq in token_encode:
        token_embeddings = []
        for token in seq:
            if token in model.wv:
                token_embeddings.append(model.wv[token])

        if token_embeddings:
            encoded_sequences.append(np.mean(token_embeddings, axis=0))
        else:
            encoded_sequences.append(np.zeros(vector_size))
   
    encoded_sequences = np.array(encoded_sequences)
                
    return encoded_sequences

  

def tokenize(text, tokenizer, merges):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])

def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(a, b, splits, word_freqs):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

def build_merges(corpus, vocab_size, tokenizer):


    word_freqs = defaultdict(int)

    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    alphabet = ['A', 'C', 'T', 'G']
    vocab = ["<|endoftext|>"] + alphabet.copy()
    splits = {word: [c for c in word] for word in word_freqs.keys()}

    pair_freqs = compute_pair_freqs(splits, word_freqs)

    best_pair = ""
    max_freq = None

    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq

    merges = {}

    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        splits = merge_pair(*best_pair, splits, word_freqs)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])

    return (merges, vocab)

def bpe(train, vocab_size):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')   

    merges, vocab = build_merges(train, vocab_size, tokenizer)

    tokens = [tokenize(seq, tokenizer, merges) for seq in train]

    return tokens

# Função para ler o CSV e gerar os encodings
def process_csv(train_file, encode_file):
    # Lê o CSV ignorando a primeira linha
    df_train = pd.read_csv(train_file)    
    df_encode = pd.read_csv(encode_file)    

    x_train = df_train['sequence']

    x_encode = df_encode['sequence']
    y_encode = df_encode['label']

    vocab_size = 50
    token_train = bpe(x_train, vocab_size)
    token_encode = bpe(x_encode, vocab_size)

    vector_size = 250
    window = 20
    min_count = 1
    epochs = 150

    encoded_sequences = w2v(token_train, token_encode, vector_size, window, min_count, epochs)

    for i in encoded_sequences:
        print(len(i))

    return encoded_sequences, y_encode


process_csv('train.csv', 'encode.csv')