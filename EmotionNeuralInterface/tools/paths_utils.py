import os

def get_paths_experiment(root_path, accepted_files=(".csv")):
  """
  Regresa las rutas donde se encuentran los archivos CSV.

  Args:
      root_path (str): Folder donde se encuentran los experimentos
      accepted_files (tuple, optional): Extensiones permitidas. Defaults to (".csv").

  Returns:
      list: Lista con todos los paths
  """  
  data = {}
  dirs = os.listdir(root_path)
  for dir in dirs:
    data_path = os.path.join(root_path,dir)
    if os.path.isdir(data_path):
      data[dir] = [os.path.join(data_path,file) for file in os.listdir(data_path) if file.lower().endswith(accepted_files)]
  return data

  