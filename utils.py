import numpy as np


def print_results(predictions, targets):
    for p, t in zip(predictions, targets):
        print(f'{p == t}\t{(p, t)}')
    print(f'accuracy: {np.mean(predictions == targets)}')
