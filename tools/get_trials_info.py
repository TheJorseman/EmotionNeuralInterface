import pandas as pd
import plotly.express as px

import argparse

"""
How to use:

python3 get_trials_info.py --file_csv <csv_file> 

Example:
    Linux:
        python3 get_trials_info.py --file_csv ../trials.csv
    Windows:
        python get_trials_info.py --file_csv ../trials.csv

"""

parser = argparse.ArgumentParser()
parser.add_argument("--file_csv", help="File yaml to convert", type=str)
args = parser.parse_args()

df = pd.read_csv(args.file_csv)
df = df[df.study_id==3]
end = pd.to_datetime(df.datetime_complete)
ini = pd.to_datetime(df.datetime_start)
df['duration'] = end - ini
df['duration_minutes'] = df['duration'].dt.total_seconds().div(60)

fig = px.line(df, y="duration_minutes", title='Duracion de los experimentos [M]')
fig.show()