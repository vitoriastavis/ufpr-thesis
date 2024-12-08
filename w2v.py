
import re
import re
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer
from collections import defaultdict
import argparse
import time

def apply_w2v(model, token_encode, vector_length):  
    """
    Applies word2vec embedding with BPE or k-mer tokenization
    in the train and eval data

    Args:
        x_train (pd.Series): Train sequences
        x_eval (pd.Series): Eval sequences
        embedding_args (dict): Contains the tokenization, window_size and vocab_size or k 

    Returns:
        encoded_train: Encoded train sequences
        encoded_eval: Encoded eval sequences
    """

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

def tokenize(text, tokenizer, merges):
    """
    Tokenizes a given text using character pair merges based on Byte Pair Encoding (BPE).
    
    Args:
        text (str): The input text to be tokenized.
        tokenizer (AutoTokenizer): A pre-trained tokenizer used for initial pre-tokenization.
        merges (dict): A dictionary containing character pairs and their merged forms.
    
    Returns:
        tokens (list): A list of tokens obtained after applying BPE.
    """

    # Pre-tokenize the sequences into words using the tokenizer's internal pre-tokenizer
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    
    # Split each word into individual characters
    splits = [[l for l in word] for word in pre_tokenized_text]

    # Apply the merges based on the provided merges
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    # Flatten list
    tokens = sum(splits, [])   

    return tokens

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
        if best_pair:
            splits = merge_pair(*best_pair, splits, word_freqs)
            merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])
        else:
            break

    return merges

def bpe(train, vocab_size):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')   

    merges = build_merges(train, vocab_size, tokenizer)

    tokens = [tokenize(seq, tokenizer, merges) for seq in train]

    return tokens

def generate_kmers(sequence, k):
    """Gera k-mers a partir de uma sequência de nucleotídeos."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def kmers(sequences, k):
    """Tokeniza uma lista de sequências em k-mers."""
    return [generate_kmers(seq, k) for seq in sequences]

def process_sequences(x_train, x_eval, embedding_args):
    """
    Tokenizes train and eval data with BPE or k-mer 
    in the 

    Args:
        x_train (pd.Series): Train sequences
        x_eval (pd.Series): Eval sequences
        embedding_args (dict): Contains the tokenization, window_size and vocab_size or k 

    Returns:
        encoded_train: Encoded train sequences
        encoded_eval: Encoded eval sequences
    """

    model_path = '~/ufpr-thesis/models-w2v/'
    vector_length = 768
    window_size = embedding_args['window_size']
    tokenization = embedding_args['tokenization'].split('-')[1]

    # Create tokens
    if tokenization == 'bpe':
        vocab_size = embedding_args['vocab_size']
        model_path = f'{model_path}{tokenization}-{vocab_size}-{window_size}'
        token_train = bpe(x_train, vocab_size)
        token_eval = bpe(x_eval, vocab_size)
    else:
        k = embedding_args['k']
        model_path = f'{model_path}{tokenization}-{k}-{window_size}'
        token_train = kmers(x_train, k)
        token_eval = kmers(x_eval, k) 

    # Load model
    model = Word2Vec.load(model_path)

    # Apply encoding to train and eval
    encoded_train = apply_w2v(model, token_train, vector_length)
    encoded_eval = apply_w2v(model, token_eval, vector_length)

    return encoded_train, encoded_eval

def train_w2v(train_path, output_path, window_size, tokenization, embedding_arg):
    """
    Trains and saves a word2vec model based on the provided parameters

    Args:
        train_path (str): Path to the input TXT file with sequences.
        output_path (str): Path to save the model
        window_size (int): Sliding window size
        tokenization (str): Represent one of the tokenizations, bpe or kmer
        embedding_arg (int): Value 

    Returns:
        None
    """

    start_time = time.time()    
    num_epochs = 250
    vector_length = 768
    min_count = 1

    df_w2v = pd.read_csv(train_path)    
    x_w2v = df_w2v.iloc[:, 0]
    
    # Create tokens
    if tokenization == 'bpe':      
        # embedding_arg = vocab_size 
        token_w2v = bpe(x_w2v, embedding_arg)
    else:
        # embedding_arg = k
        token_w2v = kmers(x_w2v, embedding_arg)

    # Train w2v
    model = Word2Vec(token_w2v, vector_size=vector_length,
                    window=window_size, min_count=min_count, sg=1)
                    
    model.train(token_w2v, total_examples=len(token_w2v), epochs=num_epochs)

    os.makedirs(output_path, exist_ok = True)
    output_path = f'{output_path}/{tokenization}-{embedding_arg}-{window_size}'
    
    model.save(output_path)

    total_time = time.time()-start_time

    with open(f'{output_path}-log.out', 'w') as f:
        f.write(f'Execution time = {round(total_time, 3)}\n')

def main():
    # Argument parsing 
    parser = argparse.ArgumentParser(description="Trains w2v")  
    parser.add_argument('-tp', '--train_path', type=str, help='Path with text to train w2v')
    parser.add_argument('-op', '--output_path', type=str, help='Path to save the models')    
 
    args = parser.parse_args()   
    output_path = args.output_path
    train_path = args.train_path

    tokenization_methods = ['bpe', 'kmer']
    vocab_sizes = [100, 200]
    kmers = [3,6]
    window_sizes = [5, 10]

    for tokenization in tokenization_methods:

        if tokenization == 'bpe':
            for vocab_size in vocab_sizes:
                for window_size in window_sizes:
                    train_w2v(train_path, output_path, window_size, tokenization, vocab_size)
        else:
            for k in kmers:
                for window_size in window_sizes:
                    train_w2v(train_path, output_path, window_size, tokenization, k)
        
if __name__ == "__main__":
    main()

