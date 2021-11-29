import pandas as pd
import json
import argparse
import sqlite3 as db
parser = argparse.ArgumentParser()
parser.add_argument("--sql_file", help="Sql File to extract info", type=str)
parser.add_argument("--output", help="Output file", type=str)
args = parser.parse_args()

"""
How to use:

python3 generate_json.py --sql_file <db_file>  --output <output_file>

Example:
    Linux:
        python3 generate_json.py --sql_file=../db.sqlite3 --output=../optuna/optuna_metadata.json
    Windows:
        python generate_json.py --sql_file=../db.sqlite3 --output=../optuna/optuna_metadata.json

"""

def get_query():
    query = """
    SELECT t.trial_id, t.number, s.study_id, s.study_name, tv.value, tp.param_name, tp.param_value
    FROM trials as t
    JOIN trial_values as tv
    ON tv.trial_id = t.trial_id
    JOIN studies as s
    ON t.study_id = s.study_id
    JOIN trial_params as tp
    ON tp.trial_id = t.trial_id
    ORDER BY tv.value DESC 
    """
    return query

def main():
    conn = db.connect(args.sql_file)
    df = pd.read_sql_query(get_query(), conn)
    ids = df['trial_id'].unique()
    output = {}
    for trial_id in ids:
        trial_df = df[df['trial_id'] == trial_id]
        name = "optuna_{}_{}.pt".format(trial_df['study_name'].iloc[0].split("_")[1], trial_id)
        output[name] = {}
        output[name]['test_accuracy'] = trial_df['value'].iloc[0]
        for row in trial_df.itertuples():
            output[name][row.param_name] = row.param_value
        if 'window_size' not in output[name]:
            output[name]['window_size'] = int(trial_df['study_name'].iloc[0].split("_")[-1])
    print(output)
    with open(args.output, 'w') as f:
        json.dump(output, f)
    
if __name__ == "__main__":
    main()
