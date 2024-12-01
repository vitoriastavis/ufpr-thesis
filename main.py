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

import dataset
import onehot
import w2v
import grover
import dnabert1
import dnabert2

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the classifier (same as in BertForSequenceClassification)
class Classifier(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob, embedding):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.embedding = embedding

    def forward(self, inputs):
        
        if self.embedding == 'w2v':           
            batch_size, _ = inputs.size()
        elif self.embedding == 'onehot':
            batch_size, _, _ = inputs.size()
        elif self.embedding == 'dnabert1' or self.embedding == 'dnabert2':
            batch_size = inputs.size()
        elif self.embedding == 'grover':
            batch_size, _ = inputs.size()
        elif self.embedding == 'nt':
            batch_size, _ = inputs.size()

        inputs = inputs.view(batch_size, -1)  
    
        # Apply dropout
        output = self.dropout(inputs)

        # Apply classifier (linear layer)
        logits = self.classifier(output)
        
        return logits

# Calculate accuracy, f1, matthews correlation, precision and recall
def calculate_metrics(predictions: np.ndarray, labels: np.ndarray):

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

def prepare_datasets(split, embedding, embedding_args):

    df_train = split['train']   
    df_eval = split['eval'] 

    x_train = df_train['sequence']
    y_train = df_train['label']

    x_eval = df_eval['sequence']
    y_eval = df_eval['label']

    # Apply one-hot encoding to x_train and x_eval 
    if embedding == 'onehot':
        encoded_train, encoded_eval = onehot.process_sequences(x_train, x_eval)

    # Apply word2vec to x_train and x_eval 
    elif embedding == 'w2v':    
        encoded_train, encoded_eval = w2v.process_sequences(x_train, x_eval,
                                                            embedding_args['model_path']) 
    # Apply grover to x_train and x_eval                                      
    elif embedding == 'grover':
        encoded_train, encoded_eval = grover.process_sequences(x_train, x_eval,
                                                                embedding_args['pooling'],
                                                                embedding_args['model_type'])                                                                                                                         
    # Apply dnabert1 to x_train and x_eval                                        
    elif embedding == 'dnabert1':
        encoded_train, encoded_eval = dnabert1.process_sequences(x_train, x_eval,
                                                                embedding_args['pooling'],
                                                                embedding_args['model_type'])                 
    # Apply dnabert2 to x_train and x_eval                                                               
    elif embedding == 'dnabert2':
        encoded_train, encoded_eval = dnabert2.process_sequences(x_train, x_eval,
                                                                embedding_args['pooling'],
                                                                embedding_args['model_type'])  

    # Create torch datasets
    train_dataset = dataset.MyDataset(encoded_train, y_train)
    eval_dataset = dataset.MyDataset(encoded_eval, y_eval)

    # Create DataLoader
    dataloader_train = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4)
    dataloader_eval = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4)

    return dataloader_train, dataloader_eval

def train(model, dataloader, num_epochs, optimizer, loss_function):

    # Set the model to training mode
    model.train()  
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

def eval(model, dataloader):
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="runs classification based on onehot, w2v, dnabert1 or dnabert2 embeddings")    
    parser.add_argument('-fp', '--file_path', type=str, help='Path to csv data')    
    parser.add_argument('-op', '--output_path', type=str, help='Output path for results')

    args = parser.parse_args()   

    # Path to inputs
    file_path = args.file_path

    # Results path    
    output_path = args.output_path

    return (file_path, output_path)

def create_splits(file_path, n_splits=3, test_size=0.2, random_seeds=None, stratify_column='label'):
    """
    Splits a CSV file into multiple train-test splits and returns them as DataFrames.

    Args:
        file_path (str): Path to the CSV file.
        n_splits (int): Number of splits to create. Default is 3.
        test_size (float): Proportion of the data to use for testing in each split. Default is 0.3.
        random_seeds (list[int]): List of random seeds for reproducibility. If None, defaults to [0, 1, 2].
        stratify_column (str): Column to use for stratification. If None, no stratification is applied.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - 'train': DataFrame with the training split.
            - 'test': DataFrame with the testing split.
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
            test_size=test_size,
            random_state=seed,
            stratify=df[stratify_column] if stratify_column else None
        )
        splits.append({'train': train_df.reset_index(drop=True), 'eval': eval_df.reset_index(drop=True)})
    
    return splits

def run(embedding, split, output_path, learning_rate, hidden_size, num_epochs, embedding_args):
 
    # CLassifier parameters
    n_labels = 2     
    dropout_prob = 0.1  

    os.makedirs(output_path, exist_ok = True)

    with open(f'{output_path}/log.out', "w") as file:

        # Create dataloaders from inputs
        dataloader_train, dataloader_eval = prepare_datasets(split, embedding, embedding_args)

        file.write('> datasets loaded\n')
        file.write(f'Embedding: {embedding}\n\n')

        if embedding != 'onehot':
            file.write(f'Embedding arguments: {embedding_args}\n\n')

        # Initialize the model, optimizer, and loss function
        model = Classifier(hidden_size, n_labels, dropout_prob, embedding)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()  

        file.write('> model loaded\n')
        file.write(f"Learning rate: {learning_rate}\n")
        file.write(f"Number of epochs: {num_epochs}\n")
        file.write(f"Hidden size: {hidden_size}\n\n")

        # Train the classifier
        model, first_loss, last_loss = train(model, dataloader_train, num_epochs, optimizer, loss_function)
        file.write('> training complete\n')
        file.write(f"Epoch 0/{num_epochs}, loss = {round(first_loss, 3)}\n")
        file.write(f"Epoch {num_epochs}/{num_epochs}, loss = {round(last_loss, 3)}\n\n")

        # Evaluation
        metrics = eval(model, dataloader_eval)

        file.write('> evaluation complete\n')
        file.write(f"Accuracy: {metrics['accuracy']}\n")
        file.write(f"Precision: {metrics['precision']}\n")
        file.write(f"Recall: {metrics['recall']}\n")
        file.write(f"F1-score: {round(metrics['f1'],2)}\n")
        file.write(f"Matthew's correlation: {metrics['matthews']}\n")                   
  
def main():
    file_path, results_path = parse_arguments()

    # all_embeddings = ['onehot', 'w2v', 'grover', 'dnabert1', 'dnabert2']
    learning_rates = [0.003, 0.0003] 
    num_epochs = [20, 100]          
    pooling_methods =  ['mean', 'max']
    # dnabert_models = ['pretrained',
    #                   'finetuned-motifs']
    # dnabert2_models = ['pretrained',
    #                   'finetuned-cancer']
    # grover_models = ['pretrained',
    #                  'finetuned-cancer']

    all_embeddings = ['dnabert1', 'dnabert2', 'grover']
    # learning_rates = [0.003] 
    # num_epochs = [20, 10]          
    # pooling_methods =  ['mean']
    dnabert1_models = ['pretrained']
    dnabert2_models = ['finetuned-cancer']
    grover_models = ['finetuned-cancer']

    w2v_path = './w2v-models'
    w2v_models = [os.path.join(w2v_path, f) for f in os.listdir(w2v_path) if os.path.isfile(os.path.join(w2v_path, f)) and f.endswith('_model')]
    
    n_splits = 3
    splits = create_splits(file_path, n_splits)

    for embedding in all_embeddings:

        # Begins counting for each embedding
        count = 25

        for i in range(n_splits):

            # Get split
            split = splits[i]

            for learning_rate in learning_rates:
                for epochs in num_epochs:
                    
                    if embedding == 'onehot':  
                        output_path = f'{results_path}/{embedding}/split{i+1}/{count}'
                        hidden_size = 404                   
                        run(embedding, split, output_path, learning_rate, hidden_size, epochs, {})
                        print(f'Done {embedding} {count}')                   
                        count += 1

                    elif embedding == 'w2v':  
                        hidden_size = 250                     
                        for model_path in w2v_models:
                            output_path = f'{results_path}/{embedding}/split{i+1}/{count}'
                            embedding_args = {'model_path': model_path}
                            run(embedding, split, output_path, learning_rate, hidden_size, epochs, embedding_args)
                            print(f'Done {embedding} {count}')
                            count += 1

                    elif embedding == 'grover':
                        hidden_size = 768
                        for pooling in pooling_methods:
                            for model_type in grover_models:
                                output_path = f'{results_path}/{embedding}/split{i+1}/{count}'
                                embedding_args = {'pooling': pooling, 'model_type': model_type}   
                                run(embedding, split, output_path, learning_rate, hidden_size, epochs, embedding_args)
                                print(f'Done {embedding} {count}')
                                count += 1

                    elif embedding == 'dnabert1':
                        hidden_size = 768
                        for pooling in pooling_methods:
                            for model_type in dnabert1_models:
                                output_path = f'{results_path}/{embedding}/split{i+1}/{count}'
                                embedding_args = {'pooling': pooling, 'model_type': model_type}   
                                run(embedding, split, output_path, learning_rate, hidden_size, epochs, embedding_args)
                                print(f'Done {embedding} {count}')
                                count += 1

                    elif embedding == 'dnabert2':
                        hidden_size = 768
                        for pooling in pooling_methods:
                            for model_type in dnabert2_models:
                                output_path = f'{results_path}/{embedding}/split{i+1}/{count}'
                                embedding_args = {'pooling': pooling, 'model_type': model_type}   
                                run(embedding, split, output_path, learning_rate, hidden_size, epochs, embedding_args)
                                print(f'Done {embedding} {count}')
                                count += 1

if __name__ == "__main__":
    main()


