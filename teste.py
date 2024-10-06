import os 
import argparse
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
    
    output_path = f'{output_path}/{n_epochs}/{learning_rate}/'	
    os.makedirs(output_path, exist_ok=True)
    # Create dataloaders from inputs
    with open(f'{output_path}/log.out', "w") as file:
      file.write(f"\tModel parameters:\n")
      file.write(f"Training data: {train_path}\n")
      file.write(f"Evaluation data: {eval_path}\n")
      file.write(f"Embedding: {embedding}\n")
      file.write(f"Learning rate: {learning_rate}")
      file.write(f"Number of epochs: {n_epochs}\n\n")

      file.write('2+2')

if __name__ == "__main__":
    main()


