import optuna
import yaml
import json
from datetime import datetime

from workbench import Workbench

DATA_YAML = "config/config.yaml"

data = yaml.load(open(DATA_YAML), Loader=yaml.FullLoader)

models = {'linear' : "config/models/linear.yaml", 'conv1d':"config/models/conv1d.yaml", 'conv2d': "config/models/stagenet.yaml", 'nedbert': "config/models/nedbert.yaml"}
key_model = 'linear'
selected_model = models[key_model]
task_list = ["same_channel_single_channel", "same_subject_single_channel", "consecutive_single_channel"]
optimizers = ["adam", "sgd"]
EPOCHS = 13

dataset_train_len = 700000
dataset_test_len = 200000
dataset_validation_len = 100000
FULL = 1
def get_batch_range(key_model):
    if key_model in ['linear', 'conv1d']:
        return 8,256
    elif key_model == 'conv2d':
        return 8,72
    else:
        return 2,6

def get_window_range(key_model):
    if key_model in 'linear':
        return 64,2048
    elif key_model == 'conv2d':
        return 128,2048
    elif key_model == 'conv2d':
        return 128,2048
    else:
        return 128,2048

def common_hyperparameters(trial, data):
    data['optimizer']['name'] = trial.suggest_categorical("optimizer", optimizers)
    data['optimizer']['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    b_i, b_e = get_batch_range(key_model)
    data['dataset_batch']['train_batch_size'] = trial.suggest_int('batch_size', b_i, b_e)
    data['tokenizer']['window_size'] = trial.suggest_int('window_size', 64, 2048, step=64)
    data['tokenizer']['stride'] =  trial.suggest_int('stride', 64, 1024, step=64)
    data['loss']['margin'] = trial.suggest_float("margin", 0.1, 10)

def objective(trial):
    common_hyperparameters(trial, data)
    data['datagen_config']['dataset'] = trial.suggest_categorical("task", task_list)
    data['train']['epochs'] = EPOCHS
    data['model']['model_config_path'] = selected_model
    data['datagen_config']['dataset_train_len'] = dataset_train_len
    data['datagen_config']['dataset_test_len'] = dataset_test_len
    data['datagen_config']['dataset_validation_len'] = dataset_validation_len
    exp = Workbench(data)
    exp.run_optuna()
    return exp.test_accuracy


storage = optuna.storages.RDBStorage(
    "sqlite:///db.sqlite3",
    heartbeat_interval=1
)
study = optuna.create_study(storage=storage, study_name="pytorch_{}".format(key_model,FULL), direction='maximize', load_if_exists=True)
study.optimize(objective, n_trials=50)
output = {'model': key_model, 'epochs': EPOCHS}
output.update(study.best_params)
with open('result-optuna-{}.json'.format(datetime.now().timestamp()), 'w') as f:
    json.dump(output, f)
