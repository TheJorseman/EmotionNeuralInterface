import os
import argparse
import json
import shutil
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
parser.add_argument('--experiments_json', type=str, required=True)

def create_folders(base_dir):
    """
    Create folders for experiments.
    """
    zs = os.path.join(base_dir, "zero_shot")
    ft = os.path.join(base_dir, "finetuning")
    zt = os.path.join(base_dir, "zero_training")
    if not os.path.exists(zs):
        os.makedirs(zs)
    if not os.path.exists(ft):
        os.makedirs(ft)
    if not os.path.exists(zt):
        os.makedirs(zt)
    return {
        "zero_shot": zs,
        "finetuning": ft,
        "zero_training": zt
    }

def parse_model_path(model_path):
    """
    Parse model path to get model name.
    """
    return os.path.basename(model_path)


def move_experiments(folders, experiments_json):
    """
    Sort experiments by their name.
    """
    with open(experiments_json, 'r') as f:
        data = json.load(f)
        for model_name,vals in data.items():
            print("Model: {}".format(model_name))
            for key,val in vals.items():
                print("{}: {}".format(key, val))
                if key == "zero_shot":
                    shutil.move(val, folders["zero_shot"])
                elif key == "finetuning":
                    shutil.move(val, folders["finetuning"])
                elif key == "zero_training":
                    shutil.move(val, folders["zero_training"])

if __name__ == '__main__':
    args = parser.parse_args()
    folders = create_folders("models")
    move_experiments(folders, args.experiments_json)