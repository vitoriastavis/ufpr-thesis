import os
import sys
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

import dataset
import onehot
import w2v
import dnabert1
import dnabert2

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

def prepare_datasets(train_path, eval_path, embedding, embedding_args):

    # Apply one-hot encoding
    if embedding == 'onehot':
        x_train, y_train, x_eval, y_eval = onehot.process_csv(train_path, eval_path)

    # Apply word2vec
    elif embedding == 'w2v':    
        x_train, y_train, x_eval, y_eval = w2v.process_csv(train_path,
                                                            eval_path,
                                                            embedding_args['model_path'] 
                                                            )                                                       
    # Apply dnabert1                                       
    elif embedding == 'dnabert1':
        x_train, y_train, x_eval, y_eval = dnabert1.process_csv(train_path,
                                                                eval_path,
                                                                embedding_args['pooling']
                                                                )                                       
    # Apply dnabert2                                                                  
    else:
        x_train, y_train, x_eval, y_eval = dnabert2.process_csv(train_path,
                                                                eval_path,
                                                                embedding_args['pooling']
                                                                )

    # Create torch datasets
    train_dataset = dataset.MyDataset(x_train, y_train)
    eval_dataset = dataset.MyDataset(x_eval, y_eval)

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
        # print(epoch)
        total_loss = 0

        # Iterate through batches 
        for batch in dataloader: 

            inputs, labels = batch  # Get inputs and labels

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
            loss_list.append(total_loss / len(dataloader))

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
    parser.add_argument('-tp', '--train_path', type=str, help='Path to train data .csv')    
    parser.add_argument('-ep', '--eval_path', type=str, help='Path to eval data .csv')    
    parser.add_argument('-op', '--output_path', type=str, help='Output path for results')

    args = parser.parse_args()   

    # Path to inputs
    train_path = args.train_path
    eval_path = args.eval_path

    # Results path    
    output_path = args.output_path

    return (train_path, eval_path, output_path)

def run(embedding, train_path, eval_path, output_path, learning_rate, hidden_size, num_epochs, embedding_args):
 
    # Parameters
    n_labels = 2     
    dropout_prob = 0.1  

    os.makedirs(output_path, exist_ok = True)

    with open(f'{output_path}/log.out', "w") as file:

        # Create dataloaders from inputs
        dataloader_train, dataloader_eval = prepare_datasets(train_path, eval_path, embedding, embedding_args)

        file.write('> datasets loaded\n')
        file.write(f'Train: {train_path}\n')
        file.write(f'Eval: {eval_path}\n')
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
        file.write(f"F1-score: {metrics['f1']}\n")
        file.write(f"Matthew's correlation: {metrics['matthews']}\n")                   

  
def main():
    train_path, eval_path, results_path = parse_arguments()

    # testar hidden size como 404 pra todos

    # all_embeddings = ['onehot', 'w2v', 'dnabert1', 'dnabert2']
    all_embeddings = ['onehot', 'dnabert1', 'dnabert2']

    # learning_rates = [0.003, 0.03, 0.0003] 
    # num_epochs = [50, 150, 200]          
    # pooling_methods =  ['mean', 'max']
    learning_rates = [0.003] 
    num_epochs = [2]          
    pooling_methods = ['mean']

    model_w2v = '/mnt/NAS/stavisa/w2v_real/models'

    # hidden_size = 404

    for embedding in all_embeddings:
        count = 1
        for learning_rate in learning_rates:
            for epochs in num_epochs:
                output_path = f'{results_path}/{embedding}/{count}'
                if embedding == 'onehot':     
                    hidden_size = 404                   
                    embedding_args = {}
                    run(embedding, train_path, eval_path, output_path, learning_rate, hidden_size, epochs, embedding_args)
                    print(f'terminei {embedding} {count}')
                   
                    count += 1
                elif embedding == 'w2v':
                    embedding_args = {'model_path':''}
                elif embedding == 'dnabert1':
                    hidden_size = 768
                    for pooling in pooling_methods:
                        embedding_args = {'pooling': pooling}   
                        run(embedding, train_path, eval_path, output_path, learning_rate, hidden_size, epochs, embedding_args)
                        count += 1
                        print(f'terminei {embedding} {count}')

                elif embedding == 'dnabert2':
                    hidden_size = 768

                    for pooling in pooling_methods:
                        embedding_args = {'pooling': pooling}   
                        run(embedding, train_path, eval_path, output_path, learning_rate, hidden_size, epochs, embedding_args)
                        count += 1
                        print(f'terminei {embedding} {count}')


if __name__ == "__main__":
    main()


