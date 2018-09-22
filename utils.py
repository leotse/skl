import numpy as np


def print_report(predictions, targets):
    print_results(predictions, targets)
    print_error(predictions, targets)
    print_accuracy(predictions, targets)


def print_results(predictions, targets):
    for p, t in zip(predictions, targets):
        print(f'{p == t}\t{(p, t)}')


def print_error(predictions, targets):
    print(f'error: {np.mean(predictions != targets)}')


def print_accuracy(predictions, targets):
    print(f'accuracy: {np.mean(predictions == targets)}')
