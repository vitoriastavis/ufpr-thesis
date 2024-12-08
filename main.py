import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from sklearn.model_selection import train_test_split
import time

import dataset
import onehot
import w2v
import grover
import dnabert1
import dnabert2

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Classifier(nn.Module):
    """
    Builds the Linear classifier
    """
    
    def __init__(self, hidden_size, num_labels, dropout_prob, embedding):
        """
        Initializes the classifier, with the dropout layer, 
        linear classifier layer, and embedding method.

        Args:
            hidden_size (int): The size of the hidden layer.
            num_labels (int): The number of output labels.
            dropout_prob (float): The dropout probability used for regularization.
            embedding (str): The embedding method to be used by the model.
        
        Returns:
            None
        """     

        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.embedding = embedding

    def forward(self, inputs):
        """
        Forward pass, applies dropout regularization, and passes the data through the classifier.

        Args:
            inputs (Tensor): The input data to be processed by the model.
        
        Returns:
            logits (Tensor): The output predictions after applying the classifier.
        """

        embedding = self.embedding.split('-')[0]
        
        # Get batch size
        if embedding == 'w2v':            
            batch_size, _ = inputs.size()
        elif embedding == 'onehot':
            batch_size, _, _ = inputs.size()
        elif embedding == 'dnabert1' or embedding == 'dnabert2':
            batch_size = inputs.size()
        elif embedding == 'grover':
            batch_size, _ = inputs.size()
 
        # Reshapes the inputs
        inputs = inputs.view(batch_size, -1)  
    
        # Apply dropout regularization
        output = self.dropout(inputs)

        # Apply classifier (linear layer)
        logits = self.classifier(output)
        
        return logits

def prepare_datasets(split, embedding, embedding_args):
    """
    Calls the embeddings methods and builds train and eval dataloaders

    Args:
        split (list): Curret split with train and eval dataframes
        embedding (str): Embedding method
        embedding_args (dict): specific arguments for each embedding method

    Returns:
        dataloader_train (DataLoader): Data structure with training dataset
        dataloader_eval (DataLoader): Data structure with evaluation dataset
    """

    df_train = split['train']   
    df_eval = split['eval'] 

    x_train = df_train['sequence']
    y_train = df_train['label']

    x_eval = df_eval['sequence']
    y_eval = df_eval['label']

    print(embedding)

    # Apply one-hot encoding to x_train and x_eval 
    if embedding == 'onehot':
        encoded_train, encoded_eval = onehot.process_sequences(x_train, x_eval)

    # Apply word2vec to x_train and x_eval 
    elif embedding == 'w2v-bpe' or embedding == 'w2v-kmer':     
        encoded_train, encoded_eval = w2v.process_sequences(x_train, x_eval,
                                                            embedding_args) 
    # Apply grover to x_train and x_eval                                      
    elif embedding == 'grover-pretrained' or embedding == 'grover-finetuned-cancer':
        encoded_train, encoded_eval = grover.process_sequences(x_train, x_eval,
                                                                embedding_args['pooling'],
                                                                embedding_args['model_type'])                                                                                                                         
    # Apply dnabert1 to x_train and x_eval                                        
    elif embedding == 'dnabert1-pretrained' or embedding == 'dnabert1-finetuned-motifs':
        encoded_train, encoded_eval = dnabert1.process_sequences(x_train, x_eval,
                                                                embedding_args['pooling'],
                                                                embedding_args['model_type'])                 
    # Apply dnabert2 to x_train and x_eval                                                               
    elif embedding == 'dnabert2-pretrained' or embedding == 'dnabert2-finetuned-cancer':
        encoded_train, encoded_eval = dnabert2.process_sequences(x_train, x_eval,
                                                                embedding_args['pooling'],
                                                                embedding_args['model_type'])  

    # Create torch datasets
    train_dataset = dataset.MyDataset(encoded_train, y_train)
    eval_dataset = dataset.MyDataset(encoded_eval, y_eval)

    # Create DataLoader from torch datasets
    dataloader_train = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4)
    dataloader_eval = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4)

    return dataloader_train, dataloader_eval

def train(model, dataloader, num_epochs, optimizer):
    """
    Performs the model training with train dataloader.

    Args:
        model (Classifier): Torch model to use as classifier
        dataloader (DataLoader): Data structure storing the training data
        num_epochs (int): Number of epochs to train the classifier
        optimizer (Adam): Adam stochastic optimizer with learning rate 

    Returns:
        model (Classifier): Trained classifier with updated weights
        loss_list[0]: Initial loss before training
        loss_list[-1]: Final loss, after training
    """

    # Set the model to training mode
    model.train()  
    loss_function = nn.CrossEntropyLoss()
    loss_list = []

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0  

        # Iterate through batches 
        for batch in dataloader: 

            # Get inputs and labels
            inputs, labels = batch  

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model.forward(inputs)

            # Compute loss
            loss = loss_function(logits, labels)            

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_list.append(total_loss/len(dataloader))

    return model, loss_list[0], loss_list[-1]


# Calculate accuracy, f1, matthews correlation, precision and recall
def calculate_metrics(predictions: np.ndarray, labels: np.ndarray):
    """
    Calculates the avaliation metrics from prediction and labels

    Args:
        predictions (np.ndarray): Array containing the eval predictions 
        labels (np.ndarray): Array containing the correct labels

    Returns:
        metrics (dict): Evaluation results (accuracy, precision, recall, F1, MCC)
    """
    metrics = {
        "accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        "f1": sklearn.metrics.f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "matthews": sklearn.metrics.matthews_corrcoef(
            labels, predictions
        ),
        "precision": sklearn.metrics.precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        }

    return metrics

def eval(model, dataloader):
    """
    Performs the model evaluation/test with eval dataloader and get metrics.

    Args:
        model (Classifier): Torch model to use as classifier
        dataloader (DataLoader): Data structure storing the evaluation data

    Returns:
        metrics (dict): Evaluation results (accuracy, precision, recall, F1, MCC)
    """

    all_predictions = []
    all_labels = []

    # Set the model to evaluation mode
    model.eval() 
    with torch.no_grad():  # Disable gradient calculation 
        
        # Iterate through the data loader
        for inputs, labels in dataloader:  

            # Get the logits from the classifier
            logits = model(inputs)

            # Predicted classes (index of the max logits)
            predicted_classes = torch.argmax(logits, dim=-1)

            # Accumulate predictions and labels
            all_predictions.extend(predicted_classes.numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_labels)

    return metrics

def create_splits(file_path, n_splits=3, eval_size=0.2, random_seeds=None, stratify_column='label'):
    """
    Splits a CSV file into multiple train-eval splits and returns them as DataFrames.

    Args:
        file_path (str): Path to the CSV file.
        n_splits (int): Number of splits to create. Default is 3.
        eval_size (float): Proportion of the data to use for evaluation in each split. Default is 0.2.
        random_seeds (list[int]): List of random seeds for reproducibility. If None, defaults to [0, 1, 2].
        stratify_column (str): Column to use for stratification. If None, no stratification is applied.

    Returns:
        splits (list[int]): A list with n_splits elements, where each element contains a dictionary with:
            - 'train_df': DataFrame with the training split.
            - 'eval_df': DataFrame with the evaluation split.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Default random seeds if not provided
    if random_seeds is None:
        random_seeds = list(range(n_splits))
    
    # Ensure the number of seeds matches the number of splits
    if len(random_seeds) != n_splits:
        raise ValueError("The number of random_seeds must match n_splits.")
    
    # Prepare a list to store splits
    splits = []
    
    # Create each split
    for seed in random_seeds:
        train_df, eval_df = train_test_split(
            df,
            test_size=eval_size,
            random_state=seed,
            stratify=df[stratify_column] if stratify_column else None
        )
        splits.append({'train': train_df.reset_index(drop=True), 'eval': eval_df.reset_index(drop=True)})
    
    return splits

def run(embedding, split, output_path, learning_rate, num_epochs, count, embedding_args):
    """
    Runs dataset preparation, model initialization,
    training, evaluation, and log results.

    Args:
        embedding (str): The embedding method to be used by the model.
        split (list): The dataset split containing training and evaluation data.
        output_path (str): The directory to save the output logs and results.
        learning_rate (float): The learning rate for the optimizer.
        num_epochs (int): The number of training epochs.
        count (int): The identifier for the current run (used for output file naming).
        embedding_args (dict): Specific arguments for the selected embedding method.
        
    Returns:
        None
    """ 

    # Classifier parameters
    n_labels = 2     
    dropout_prob = 0.1  
    
    if embedding == 'onehot':
        hidden_size = 404
    else:
        hidden_size = 768

    os.makedirs(output_path, exist_ok = True)

    with open(f'{output_path}/{count}.out', "w") as file:

        # Create dataloaders from inputs
        start_time = time.time()
        dataloader_train, dataloader_eval = prepare_datasets(split, embedding, embedding_args)
        total_time = time.time()-start_time

        file.write('> datasets loaded\n')
        file.write(f'Embedding {embedding} took {round(total_time, 3)}s\n')
        file.write('Embedding args:\n')

        # if embedding != 'onehot':
        #     for k, v in embedding_args.items():
        #         if k != 'tokenization':
        #             file.write(f'{k}: {v}; ')
        # else:
        #     file.write('\n')    

        # # Initialize the model, optimizer, and loss function
        # model = Classifier(hidden_size, n_labels, dropout_prob, embedding)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)  

        # file.write('\n\n> model loaded\n')
        # file.write(f"Learning rate: {learning_rate}\n")
        # file.write(f"Number of epochs: {num_epochs}\n")
        # file.write(f"Hidden size: {hidden_size}\n\n")

        # # Train the classifier
        # model, first_loss, last_loss = train(model, dataloader_train, num_epochs, optimizer)
        # file.write('> training complete\n')
        # file.write(f"Epoch 0/{num_epochs}, loss = {round(first_loss, 3)}\n")
        # file.write(f"Epoch {num_epochs}/{num_epochs}, loss = {round(last_loss, 3)}\n\n")

        # # Evaluation
        # metrics = eval(model, dataloader_eval)

        # file.write('> evaluation complete\n')
        # file.write(f"Accuracy: {round(metrics['accuracy'], 3)}\n")
        # file.write(f"Precision: {round(metrics['precision'], 3)}\n")
        # file.write(f"Recall: {round(metrics['recall'], 3)}\n")
        # file.write(f"F1-score: {round(metrics['f1'], 3)}\n")
        # file.write(f"Matthew's correlation: {round(metrics['matthews'], 3)}\n")   

def parse_arguments():
    """
    Get arguments from input and parse them into the correct variables

    Args:
        None

    Returns:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to the output directory.
    """
    parser = argparse.ArgumentParser(description="runs classification based on onehot, w2v, dnabert1 or dnabert2 embeddings")    
    parser.add_argument('-fp', '--file_path', type=str, help='Path to csv data')    
    parser.add_argument('-op', '--output_path', type=str, help='Output path for results')

    args = parser.parse_args()   

    # Path to inputs
    file_path = args.file_path

    # Results path    
    output_path = args.output_path

    return (file_path, output_path)                
  
def main():
    """
    Sets embedding names, embedding arguments, classifier parameters,
    create splits and calls the run() function
    """
    file_path, results_path = parse_arguments()

    all_embeddings = ['w2v-bpe', 'w2v-kmer',
                      'dnabert1-pretrained', 'dnabert1-finetuned-motifs',
                      'dnabert2-pretrained', 'dnabert2-finetuned-cancer',
                      'grover-pretrained', 'grover-finetuned-cancer']

    # Classifier parameters
    learning_rates = [0.003, 0.0003] 
    num_epochs = [20, 100]             

    # Embedding parameters
    # pooling_methods =  ['mean', 'max']
    pooling_methods =  ['mean']
    vocab_sizes = [100, 200]
    kmers = [3, 6]
    window_sizes = [5, 10]

    # Create splits
    n_splits = 1
    splits = create_splits(file_path, n_splits)

    for embedding in all_embeddings:

        # Begins each embedding in model #1
        count = 1

        for i in range(n_splits):

            # Get split
            split = splits[i]
            output_path = f'{results_path}/{embedding}/split{i+1}'

            for learning_rate in learning_rates:
                for epochs in num_epochs:
                    
                    if embedding == 'onehot':                                          
                        run(embedding, split, output_path, learning_rate, epochs, count, {})
                        print(f'Done {embedding} {count}')                   
                        count += 1

                    if embedding == 'w2v-bpe':  
                        for vocab_size in vocab_sizes:
                            for window_size in window_sizes:
                                embedding_args = {'vocab_size': vocab_size, 'window_size': window_size, 'tokenization': embedding}
                                run(embedding, split, output_path, learning_rate, epochs, count, embedding_args)
                                print(f'Done {embedding} {count}')
                                count += 1

                    elif embedding == 'w2v-kmer':  
                        for k in kmers:
                            for window_size in window_sizes:
                                embedding_args = {'k': k, 'window_size': window_size, 'tokenization': embedding}
                                run(embedding, split, output_path, learning_rate, epochs, count, embedding_args)
                                print(f'Done {embedding} {count}')
                                count += 1

                    else:
                        for pooling in pooling_methods:
                            embedding_args = {'pooling': pooling, 'model_type': embedding}   
                            run(embedding, split, output_path, learning_rate, epochs, count, embedding_args)
                            print(f'Done {embedding} {count}')
                            count += 1                        

if __name__ == "__main__":
    main()


