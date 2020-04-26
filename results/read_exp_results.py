import json
import ast

import os
import pandas as pd
import numpy as np


def read_exp_results(path, filename):
    files = os.listdir(path)
    results = []
    for exp in files:    
        with open(path + exp) as file:
            x = ast.literal_eval(file.readlines()[0])
            x = ast.literal_eval('{' + x[1:-1] + '}')
            results.append(x)
        
    loss = np.array([x['loss'] for x in results])
    val_loss = np.array([x['val_loss'] for x in results])
    accuracy = np.array([x['accuracy'] for x in results])
    val_accuracy = np.array([x['val_accuracy'] for x in results])
    
    output = pd.DataFrame({'loss_mean': loss.mean(axis=0),
                     'loss_std': loss.std(axis=0),
                      'val_loss_mean': val_loss.mean(axis=0),
                     'val_loss_std': val_loss.std(axis=0),
                      'accuracy_mean': accuracy.mean(axis=0),
                     'accuracy_std': accuracy.std(axis=0),
                      'val_accuracy_mean': val_accuracy.mean(axis=0),
                     'val_accuracy_std': val_accuracy.std(axis=0),
                     })

    output.to_csv(filename)
    
    return output

def main():
    print('Architecture 1:')
    print(read_exp_results('./architecture-1/', filename = 'architecture-1-results.csv'))

    print('Architecture 2:')
    print(read_exp_results('./architecture-2/', filename = 'architecture-2-results.csv'))

    print('Architecture 3:')
    print(read_exp_results('./architecture-3/', filename = 'architecture-3-results.csv'))

    print('Architecture 4:')
    print(read_exp_results('./architecture-4/', filename = 'architecture-4-results.csv'))

if __name__ == "__main__":
    main()

