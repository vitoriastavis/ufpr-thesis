import nltk
import urllib
import sys
import re
import bs4 as bs
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict
import argparse

# Aplica o word2vec nas sequencias 
def apply_w2v(model, token_encode, vector_length):  

    encoded_sequences = []

    for seq in token_encode:
        token_embeddings = []
        for token in seq:
            if token in model.wv:
                token_embeddings.append(model.wv[token])

        if token_embeddings:
            encoded_sequences.append(np.mean(token_embeddings, axis=0))
        else:
            encoded_sequences.append(np.zeros(vector_length))
   
    encoded_sequences = np.array(encoded_sequences)
                
    return encoded_sequences


def train_w2v(train_path, vocab_size, window_size, num_epochs, vector_length, save_path):
    df_w2v = pd.read_csv(train_path)    
    x_w2v = df_w2v.iloc[:, 0]
    
    # Create BPE tokens
    token_w2v = bpe(x_w2v, vocab_size)

    # Train w2v
    model = Word2Vec(token_w2v, vector_size=vector_length,
                    window=window_size, min_count=min_count, sg=1)
    model.train(token_w2v, total_examples=len(token_w2v), epochs=num_epochs)

    model.save(save_path)

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

    return merges

def bpe(train, vocab_size):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')   

    merges = build_merges(train, vocab_size, tokenizer)

    tokens = [tokenize(seq, tokenizer, merges) for seq in train]

    return tokens

# Função para ler o CSV e gerar os encodings
def process_csv(train_file, eval_file, model_path):

    # Get the vocab size that matches the trained model
    match = re.search(r"models/(\d+)_(\d+)_(\d+)_model", model_path)
    if match:
        vocab_size = int(match.group(1))  # Extract vocab_size
    else:
        raise ValueError("Invalid model path format. Ensure it follows the structure: models/{vocab_size}_{window_size}_{epoch}_model")

    # Carregar o modelo treinado do caminho especificado
    model = Word2Vec.load(model_path)

    # Read files for training and eval
    df_train = pd.read_csv(train_file)    
    df_eval = pd.read_csv(eval_file) 

    x_train = df_train['sequence']
    y_train = df_eval['label']

    x_eval = df_train['sequence']
    y_eval = df_eval['label']
    
    # Create BPE tokens
    token_train = bpe(x_train, vocab_size)
    token_eval = bpe(x_eval, vocab_size)

    # Apply encoding to train and eval
    encoded_train = apply_w2v(model, token_encode, vector_size)
    encoded_eval = apply_w2v(model, token_encode, vector_size)

    return encoded_train, y_train, encoded_eval, y_eval

def main():
    # Argument parsing 
    parser = argparse.ArgumentParser(description="trains or applies w2v")    
    parser.add_argument('-tp', '--train_path', type=str, help='Path with text to train w2v')    
    parser.add_argument('-vs', '--vocab_size', type=int, help='Size of vocabulary') 
    parser.add_argument('-ws', '--window_size', type=int, help='Sliding window size') 
    parser.add_argument('-ne', '--num_epochs', type=int, help='Number of epochs') 
    parser.add_argument('-vl', '--vector_length', type=int, help='Length of the vector to represent each word') 
    parser.add_argument('-op', '--save_path', type=str, help='Path with text to train w2v')    

    args = parser.parse_args()   

    train_path = args.train_path
    vocab_size = args.vocab_size
    window_size = args.window_size
    num_epochs = args.num_epochs
    vector_length = args.vector_length
    save_path = args.save_path

    train_w2v(train_path, vocab_size, window_size, num_epochs, vector_length, save_path)

if __name__ == "__main__":
    main()

