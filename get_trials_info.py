import pandas as pd
import plotly.express as px
df = pd.read_csv("trials.csv")
df = df[df.study_id==3]
end = pd.to_datetime(df.datetime_complete)
ini = pd.to_datetime(df.datetime_start)
df['duration'] = end - ini
df['duration_minutes'] = df['duration'].dt.total_seconds().div(60)
fig = px.line(df, y="duration_minutes", title='Duracion de los experimentos [M]')
fig.show()