import os
import sys
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics
import torch.optim as optim
import dataset
import onehot
import w2v2
from torch.utils.data import DataLoader
import argparse
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

def prepare_datasets(train_path, eval_path, embedding, args):

    # Apply one-hot encoding
    if embedding == 'onehot':
        x_train, y_train = onehot.process_csv(train_path)
        x_eval, y_eval = onehot.process_csv(eval_path)
    # Apply word2vec
    elif embedding == 'w2v':    
        x_train, y_train, x_eval, y_eval = w2v.process_csv(args['train_w2v'],
                                                            args['vocab_size_w2v'],
                                                            args['window_size_w2v'],
                                                            args['epochs_w2v'],
                                                            train_path,
                                                            eval_path)
    elif embedding == 'dnabert1':
        print('not yet')    
    else:
        x_train, y_train, x_eval, y_eval = dnabert2.process_csv(args['pooling_dnabert2'],
                                                                train_path,
                                                                eval_path)

    # Create torch datasets
    train_dataset = dataset.MyDataset(x_train, y_train)
    eval_dataset = dataset.MyDataset(x_eval, y_eval)

    # Create DataLoader
    dataloader_train = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    dataloader_eval = DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=4)

    return dataloader_train, dataloader_eval


def train(model, dataloader, n_epochs, optimizer, loss_function):

    # Set the model to training mode
    model.train()  
    loss_list = []

    # Training loop
    for epoch in range(n_epochs):

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

def parse_arguments(args_file):  

    args = {}

    with open(args_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            args[key] = value

    train_path = args['train_path']    
    eval_path = args['eval_path']
    learning_rate = float(args['learning_rate'])
    hidden_size = int(args['hidden_size'])
    n_epochs = int(args['n_epochs'])
    output_path = args['output_path']
    embedding = args['embedding']

    embedding_args = {}
    try:
        if embedding == 'w2v':
            embedding_args['train_w2v'] = args['train_w2v']
            embedding_args['vocab_size_w2v'] = args['vocab_size_w2v']
            embedding_args['window_size_w2v'] = args['window_size_w2v']
            embedding_args['epochs_w2v'] = args['epochs_w2v']
        
        elif embedding == 'onehot':
            pass
        
        elif embedding == 'dnabert1':
            # Adicione os argumentos relacionados ao 'dnabert1' se necessário
            pass
        
        elif embedding == 'dnabert2':
            pooling_dnabert2 = args['pooling_dnabert2']
        
        else:
            raise TypeError(f"embedding must be 'w2v', 'onehot', 'dnabert1', or 'dnabert2'")

    except KeyError as e:
        print(f"Error: Missing argument '{e.args[0]}' in the input arguments file. Please check the args.txt file and the structure in the README.")

    return (train_path, eval_path, learning_rate, hidden_size, n_epochs, output_path, embedding, embedding_args)
  
def main():
    # Argument parsing 
    train_path, eval_path, learning_rate, hidden_size, n_epochs, output_path, embedding, embedding_args = parse_arguments(sys.argv[1])

    # Parameters
    n_labels = 2     
    dropout_prob = 0.1  # Same dropout as in the original DNABERT classifier

    os.makedirs(output_path, exist_ok = True)

    with open(f'{output_path}/log.out', "w") as file:

        # Create dataloaders from inputs
        dataloader_train, dataloader_eval = prepare_datasets(train_path, eval_path, embedding, embedding_args)
        file.write('> datasets loaded\n')
        file.write(f'Train: {train_path}\n')
        file.write(f'Eval: {eval_path}\n')
        file.write(f'Embedding: {embedding}\n')
        file.write(f'Embedding arguments: {embedding_args}\n\n')

        # Initialize the model, optimizer, and loss function
        model = Classifier(hidden_size, n_labels, dropout_prob, embedding)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()   
        file.write('> model loaded\n\n')

        file.write('\tModel parameters:\n')
        file.write(f"Training data: {train_path}\n")
        file.write(f"Evaluation data: {eval_path}\n")
        file.write(f"Embedding: {embedding}\n")
        file.write(f"Learning rate: {learning_rate}\n")
        file.write(f"Number of epochs: {n_epochs}\n\n")

        model, first_loss, last_loss = train(model, dataloader_train, n_epochs, optimizer, loss_function)
        file.write('> training complete\n')
        file.write(f"Epoch 0/{n_epochs}, loss = {first_loss}\n")
        file.write(f"Epoch {n_epochs}/{n_epochs}, loss = {last_loss}\n\n")

        metrics = eval(model, dataloader_eval)
        file.write('> evaluation complete\n\n')    

        file.write('\tMetrics:\n')
        file.write(f"Accuracy: {metrics['accuracy']}\n")
        file.write(f"Precision: {metrics['precision']}\n")
        file.write(f"Recall: {metrics['recall']}\n")
        file.write(f"F1-score: {metrics['f1']}\n")
        file.write(f"Matthew's correlation: {metrics['matthews']}\n")                   

if __name__ == "__main__":
    main()


