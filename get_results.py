import os
import pandas as pd

# Directories for each method
embeddings = ['w2v', 'dnabert1', 'dnabert2', 'grover']

# embeddings = ['onehot']

# Dictionary to store all data for each method
all_data = {embedding: {} for embedding in embeddings}
# Dictionary to store the best parameters and metrics for each method
best_metrics = {embedding: {'acc': 0, 'f1': 0, 'mcc': 0, 'params': {}} for embedding in embeddings}

# Path to the main directory containing method directories
base_path = 'results/'

# Iterate through each method
for embedding in embeddings:
    method_path = os.path.join(base_path, embedding)
    
    # Iterate through each numbered subdirectory in the method directory
    for sub_dir in os.listdir(method_path):
        sub_dir_path = os.path.join(method_path, sub_dir)
        
        if os.path.isdir(sub_dir_path):
            log_path = os.path.join(sub_dir_path, 'log.out')
            # print(log_path)
            
            # Check if log.out exists
            if os.path.exists(log_path):
                with open(log_path, 'r') as log_file:
                    lines = log_file.readlines()
                    
                    # Extract parameters and metrics from the log file
                    embedding_args = lines[5].split(': ')[1].strip()
                    learning_rate = float(lines[8].split(': ')[1].strip())
                    num_epochs = int(lines[9].split(': ')[1].strip())
                    hidden_size = int(lines[10].split(': ')[1].strip())
                    acc = round(float(lines[17].split(': ')[1].strip()),2)
                    f1 = round(float(lines[20].split(': ')[1].strip()),2)
                    mcc = round(float(lines[21].split(': ')[1].strip()),2)
                    
                    # Save data in the all_data dictionary
                    data = {
                        'embedding_args': embedding_args,
                        'learning_rate': learning_rate,
                        'num_epochs': num_epochs,
                        'hidden_size': hidden_size,
                        'acc': acc,            
                        'f1': f1,
                        'mcc': mcc
                    }
                    
                    all_data[embedding][sub_dir] = data

                    # Check if this experiment has the best accuracy for the method
                    if acc > best_metrics[embedding]['acc'] and f1 > best_metrics[embedding]['f1']:
                        best_metrics[embedding]['acc'] = acc
                        best_metrics[embedding]['f1'] = f1
                        best_metrics[embedding]['mcc'] = mcc
                        data_no_metrics = updated_dict = {k: v for k, v in data.items() if k != 'acc' and k != 'f1' and k != 'mcc'}
                        best_metrics[embedding]['params'] = data_no_metrics

# # Print or save the dictionaries as needed
print("All Data:")
for k, v in all_data.items():
    for z, w in v.items(): 
        print(f'{k} ({z}): {w} \n')

print("\nBest Metrics:")

for k, v in best_metrics.items():
    print(f'{k}: {v} \n')

# Create a DataFrame to display the best results
results_table = []
for embedding, data in best_metrics.items():
    row = {
        'Embedding': embedding,
        'Accuracy': data['acc'],
        'F1 Score': data['f1'],
        'MCC': data['mcc'],
        'Learning Rate': data['params'].get('learning_rate', 'N/A'),
        'Num Epochs': data['params'].get('num_epochs', 'N/A'),
        'Hidden Size': data['params'].get('hidden_size', 'N/A'),
    }
    results_table.append(row)

# Convert to DataFrame and display
df = pd.DataFrame(results_table)
# print(df)

# Optionally save to CSV or display as needed
df.to_csv('results/best_metrics_summary.csv', index=False)