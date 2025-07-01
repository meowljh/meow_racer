import optuna
import torch
import torch.nn as nn

## to start the optimization, create a study object and pass the objective function to method optimize() ##
'''
1. attributes of the study object
    - best_value
    - best_trial
    - best_params (dict)
'''
def objective_quadratic(trial):
    '''
    @trial: Trial object that corresponds to a single execution of the objective function to obtain parameters for a trial (single call of the objective function)
    '''
    x = trial.suggest_float("x", -10., 10.) # selects parameters uniformly within the range provided
    return (x-2) ** 2

def create_nn(trial, in_size):
    '''모델 생성할 때도 trial object에서 임의로 parameter들을 랜덤하게 샘플링하면 그걸 바탕으로 모델 설계하면 됨.'''
    n_layers = trial.suggest_int("n_layers", 1, 3)
def objective_nn(trial):
    # categorical parameters #
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    # integer parameters #
    num_layers = trial.suggest_int("num_layers", 1, 3)
    # float parameters #
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 1e-4)
    
# study = optuna.create_study() # returns a study object (optimization session - a set of trials)

