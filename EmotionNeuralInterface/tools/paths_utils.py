import os

def get_paths_experiment(root_path, accepted_files=(".csv")):
  data = {}
  dirs = os.listdir(root_path)
  for dir in dirs:
    data_path = os.path.join(root_path,dir)
    if os.path.isdir(data_path):
      data[dir] = [os.path.join(data_path,file) for file in os.listdir(data_path) if file.lower().endswith(accepted_files)]
  return data

  