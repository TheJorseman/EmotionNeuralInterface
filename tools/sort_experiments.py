import os
import argparse
import yaml
import json
"""
How to use:

python3 sort_experiments.py --experiments_dir=../experiments/ --output=../optuna/optuna_metadata.json

Example:
    Linux:
        python3 sort_experiments.py --experiments_dir=models --output=experiment_dir.json
    Windows:
        python sort_experiments.py --experiments_dir=models --output=experiment_dir.json

"""

parser = argparse.ArgumentParser()
parser.add_argument('--experiments_dir', type=str, required=True)
parser.add_argument('--output', type=str, required=True)

def parse_model_path(model_path):
    """
    Parse model path to get model name.
    """
    return os.path.basename(model_path)


def sort_experiments(experiments_dir):
    """
    Sort experiments by their name.
    """
    output = {}
    experiments = filter(lambda e: 'model-classificator' in e, os.listdir(experiments_dir))
    for experiment in experiments:
        folder_name = os.path.join(experiments_dir, experiment)
        data_path = os.path.join(folder_name, "data.yaml")
        with open(data_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            model_name = parse_model_path(data['load_model']['path'])
            print(model_name)
            if model_name not in output:
                output[model_name] = {}
            if not 'only_test' in data:
                continue
            if data['only_test']:
                print("zero-shot")
                output[model_name]['zero_shot'] = folder_name
                #output[model_name]['zero_shot']['foldername'] = folder_name
            elif not data['only_test'] and data['use_pretrained']:
                print("Fine-tuning")
                output[model_name]['finetuning'] = folder_name
                #output[model_name]['finetuning']['foldername'] = folder_name
            elif not data['only_test'] and not data['use_pretrained']:
                print("zero-trained")
                output[model_name]['zero_training'] = folder_name
                #output[model_name]['zero_training']['foldername'] = folder_name
    return output

if __name__ == '__main__':
    args = parser.parse_args()
    experiments = sort_experiments(args.experiments_dir)
    with open(args.output, 'w') as f:
        json.dump(experiments, f, indent=4)