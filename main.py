import os
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics
import torch.optim as optim
import dataset
import onehot
import w2v
from torch.utils.data import DataLoader
import argparse

# Define the classifier (same as in BertForSequenceClassification)
class Classifier(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, inputs, embedding):
        
        if embedding == 'w2v':
           
            batch_size, _ = inputs.size()
        elif embedding == 'onehot':
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

def prepare_datasets(train_path, eval_path, embedding):

  # Apply one-hot encoding
  if embedding == 'onehot':
    x_train, y_train = onehot.process_csv(train_path)
    x_eval, y_eval = onehot.process_csv(eval_path)
  # Apply word2vec
  elif embedding == 'w2v':
    
    x_train, y_train = w2v.process_csv(train_path)
    x_eval, y_eval = w2v.process_csv(eval_path)    

  # print(f"x_train shape: {x_train.shape}")
  # print(f"x_eval shape: {x_eval.shape}")

  # print(f"type: {type(x_train)}")

  # Create torch DataSets
  train_dataset = dataset.MyDataset(x_train, y_train)
  eval_dataset = dataset.MyDataset(x_eval, y_eval)

  # Create DataLoader
  dataloader_train = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
  dataloader_eval = DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=4)

  return dataloader_train, dataloader_eval


def train(model, dataloader, n_epochs, optimizer, loss_function, embedding):

    # Set the model to training mode
    model.train()  

    # Training loop
    for epoch in range(n_epochs):

        total_loss = 0
        
        # Iterate through batches 
        for batch in dataloader:  
            inputs, labels = batch  # Get inputs and labels

            # print(f"input shape {inputs.shape}")
            # print(inputs)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model.forward(inputs, embedding)

            # Compute loss
            loss = loss_function(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {total_loss / len(dataloader):.4f}")

    return model


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
    parser = argparse.ArgumentParser(description="runs w2v or one-hot classification")    
    parser.add_argument('-tp', '--train_path', type=str, help='Path to train data.csv or .txt')    
    parser.add_argument('-ep', '--eval_path', type=str, help='Path to eval data.csv or .txt')    
    parser.add_argument('-em', '--embedding', type=str, help='Embedding method: w2v or onehot')   
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=0.001)    
    parser.add_argument('-ne', '--n_epochs', type=int, help='Number of epochs')
    #parser.add_argument('-bs', '--batch_size', type=int, help='Batch size')
    parser.add_argument('-op', '--output_path', type=str, help='Output path')

    args = parser.parse_args()   

    # Path to inputs
    train_path = args.train_path
    eval_path = args.eval_path

    # Word2vec or One-hot encoding
    embedding = args.embedding

    if embedding != 'w2v' and embedding != 'onehot':
        raise TypeError("Error: --embedding must be 'w2v' or 'onehot'")


    ### testar caminhos ###

    # Hyperparameters
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs
    # batch_size = args[batch_size]

    output_path = args.output_path
                       
    return (train_path, eval_path, embedding, learning_rate, n_epochs, output_path)


  
def main():
    # Set up argument parsing
 
    train_path, eval_path, embedding, learning_rate, n_epochs, output_path = parse_arguments()

    # Parameters
    hidden_size = 404  
    n_labels = 2     
    dropout_prob = 0.1  # Same dropout as in the original DNABERT classifier

    output_path = f'{output_path}/{n_epochs}/{learning_rate}'
    os.makedirs(output_path, exist_ok = True)

    with open(f'{output_path}/log.out', "w") as file:
      # Create dataloaders from inputs
      dataloader_train, dataloader_eval = prepare_datasets(train_path, eval_path, embedding)
      file.write('> datasets loaded\n')
    
      # Initialize the model, optimizer, and loss function
      model = Classifier(hidden_size, n_labels, dropout_prob)
      optimizer = optim.Adam(model.parameters(), lr=learning_rate)
      loss_function = nn.CrossEntropyLoss()   
      file.write('> model loaded\n\n')

      file.write('\tModel parameters:\n')
      file.write(f"Training data: {train_path}\n")
      file.write(f"Evaluation data: {eval_path}\n")
      file.write(f"Embedding: {embedding}\n")
      file.write(f"Learning rate: {learning_rate}\n")
      file.write(f"Number of epochs: {n_epochs}\n\n")

      model = train(model, dataloader_train, n_epochs, optimizer, loss_function, embedding)
      file.write('> training complete\n')

      metrics = eval(model, dataloader_eval)
      file.write('> evaluation complete\n\n')    

      file.write('Metrics:')
      file.write(f"Accuracy: {metrics['accuracy']}")
      file.write(f"Precision: {metrics['precision']}")
      file.write(f"Recall: {metrics['recall']}")
      file.write(f"F1-score: {metrics['f1']}")
      file.write(f"Matthew's correlation: {metrics['matthews']}")                   
 
if __name__ == "__main__":
    main()


