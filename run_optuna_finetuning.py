import optuna
import yaml
import json
import os
from datetime import datetime

from finetuning import FinetuneModel

DATA_YAML = "config/config_finetuning.yaml"

data = yaml.load(open(DATA_YAML), Loader=yaml.FullLoader)

models = {'linear' : "config/models/linear.yaml", 'conv1d':"config/models/conv1d.yaml", 'conv2d': "config/models/stagenet.yaml", 'nedbert': "config/models/nedbert.yaml"}
key_model = 'conv1d'
selected_model = models[key_model]
task_list = ["same_channel_single_channel", "same_subject_single_channel", "consecutive_single_channel"]
optimizers = ["adam", "sgd"]
EPOCHS = 40

folder_models = "optuna"

files = tuple([file for file in os.listdir(folder_models) if file.endswith(".pt")])

f = open("config/models/classificator.yaml")
data_model = yaml.load(f, Loader=yaml.FullLoader)

dataset_train_len = 700000
dataset_test_len = 150000
dataset_validation_len = 100000
FULL = 1

WINDOW_SIZE = 256

def get_batch_range(key_model, window_size):
    if key_model in ['linear', 'conv1d']:
        return 32,256
    elif key_model == 'conv2d':
        return 2,int(55000/window_size)
    else:
        return 2,6

def get_window_range(trial, key_model):
    if key_model in 'linear':
        return trial.suggest_int('window_size', 64, 2048, step=32)
    elif key_model == 'conv1d':
        return trial.suggest_int('window_size', 240, 2048, step=32)
    elif key_model == 'conv2d':
        return WINDOW_SIZE
    else:
        return 128,2048

def get_task(trial, key_model):
    if key_model in ['linear', 'conv1d']:
        return trial.suggest_categorical("task", task_list)
    else:
        return "relative_positioning_multiple_channel"

def get_embedding_dim(trial, key_model, data_model):
    if key_model in ['linear','conv1d', 'conv2d']:
        layers = [key for key in data_model['layers'].keys() if 'linear' in key]
        last_key = sorted(layers)[-1]
        data_model['layers'][last_key]['output_dim'] = trial.suggest_int('embedding_dim', 64, 1024, step=64)

def common_hyperparameters(trial, data):
    data['optimizer']['name'] = trial.suggest_categorical("optimizer", optimizers)
    data['optimizer']['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 0.7, log=True)
    data['tokenizer']['window_size'] = get_window_range(trial, key_model)
    data['load_model']['path'] = os.path.join(folder_models, trial.suggest_categorical("model", files))
    b_i, b_e = get_batch_range(key_model, data['tokenizer']['window_size'])
    data['dataset_batch']['train_batch_size'] = trial.suggest_int('batch_size', b_i, b_e)
    data['tokenizer']['stride'] =  trial.suggest_int('stride', 64, 1024, step=64)
    #data['loss']['margin'] = trial.suggest_float("margin", 0.1, 10)
    #data['use_pretrained'] = bool(trial.suggest_int('use_pretrained', 0, 1))
    data['use_pretrained'] = True
    data['only_test'] = True
    #data['only_test'] = bool(trial.suggest_int('use_pretrained', 0, 1))
    #get_embedding_dim(trial, key_model, data_model)
    if data['only_test']:
        data_model['use_model'] = False
    else:
        data_model['use_model'] = selected_model
    data['model']['model_config_path'] = data_model


def objective(trial):
    common_hyperparameters(trial, data)
    #data['datagen_config']['dataset'] = get_task(trial, key_model)
    data['train']['epochs'] = EPOCHS
    #data['model']['model_config_path'] = selected_model
    #data['datagen_config']['dataset_train_len'] = dataset_train_len
    #data['datagen_config']['dataset_test_len'] = dataset_test_len
    #data['datagen_config']['dataset_validation_len'] = dataset_validation_len
    exp = FinetuneModel(data)
    exp.run()
    exp.save_model("optuna", name="optuna_{}_{}".format(key_model, trial._trial_id))
    trial.set_user_attr("train_accuracy", exp.train_accuracy)
    trial.set_user_attr("evaluation_accuracy", exp.evaluation_accuracy)
    #trial.set_user_attr("dA_train", exp.dA_train)
    #trial.set_user_attr("dA_eval", exp.dA_eval)
    return exp.test_accuracy


storage = optuna.storages.RDBStorage(
    "sqlite:///db.sqlite3",
    heartbeat_interval=1
)

def test_all_zero_shots():
    result = {}
    data['use_pretrained'] = True
    data['only_test'] = True
    if data['only_test']:
        data_model['use_model'] = False
    else:
        data_model['use_model'] = selected_model
    data['model']['model_config_path'] = data_model
    data['train']['epochs'] = EPOCHS
    i = 1
    for model in files:
        try:
            data['load_model']['path'] = os.path.join(folder_models, model)
            exp = FinetuneModel(data)
            exp.run()
            result[model] = exp.knn_data
            with open('result-zero-shot.json', 'w') as f:
                json.dump(result, f)
        except Exception as e:
            print(e)
        i += 1
        print("{} de {}".format(i, len(files)))
    #exp.save_model("optuna", name="optuna_{}_{}".format(key_model, trial._trial_id))
    #trial.set_user_attr("train_accuracy", exp.train_accuracy)
    #trial.set_user_attr("evaluation_accuracy", exp.evaluation_accuracy)


def test_finetuning():
    #import pdb;pdb.set_trace()
    filename = 'result-finetuning-deep.json'
    result = {}
    data = yaml.load(open(DATA_YAML), Loader=yaml.FullLoader)
    data['use_pretrained'] = True
    data['only_test'] = False
    data['train']['epochs'] = EPOCHS
    if data['only_test']:
        data_model['use_model'] = False
    else:
        data_model['use_model'] = selected_model
    data['model']['model_config_path'] = data_model
    i = 1
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    for model in files:
        if model in result:
            continue
        try:
            data['load_model']['path'] = os.path.join(folder_models, model)
            exp = FinetuneModel(data)
            exp.run()
            result[model] = {}
            result[model]['knn'] = exp.knn_data
            result[model]['train_acc'] = exp.train_accuracy
            result[model]['test_acc'] = exp.test_accuracy
            with open(filename, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            print(e)
        i += 1
        print("{} de {}".format(i, len(files)))


if __name__ == '__main__':
    #test_all_zero_shots()
    test_finetuning()
    #study = optuna.create_study(storage=storage, study_name="pytorch_finetuning_{}_{}".format(key_model, WINDOW_SIZE), direction='maximize', load_if_exists=True)
    #study.optimize(objective, n_trials=50)
    #output = {'model': key_model, 'epochs': EPOCHS}
    #output.update(study.best_params)
    #with open('result-optuna-{}.json'.format(datetime.now().timestamp()), 'w') as f:
    #    json.dump(output, f)
