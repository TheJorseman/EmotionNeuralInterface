import os
import re
import shutil
from google_drive_downloader import GoogleDriveDownloader as gdd

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

def get_path(data_str):
  regex = re.compile(
      r"(\w+://)?"                # protocol                      (optional)
      r"(\w+\.)?"                 # host                          (optional)
      r"((\w+)\.(\w+))"           # domain
      r"(\.\w+)*"                 # top-level domain              (optional, can have > 1)
      r"([\w\-\._\~/]*)*(?<!\.)"  # path, params, anchors, etc.   (optional)
  )
  if regex.match(data_str):
    return download_gdrive_data(data_str)
  return data_str

def download_gdrive_data(url, default_data_folder="./data"):
  if data_exist(default_data_folder):
    return default_data_folder
  os.mkdir(default_data_folder)
  filename = "experimento.zip"
  gdd.download_file_from_google_drive(file_id=url.split("/")[5],
                                      dest_path=os.path.join(default_data_folder, filename),
                                      unzip=True)
  return default_data_folder

def data_exist(path):
  if not os.path.exists(path):
    return False
  file_list = get_paths_experiment(path)
  if len(file_list) > 0:
    return True
  shutil.rmtree(path)
  return False
  