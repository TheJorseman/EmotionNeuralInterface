import os
from .data_parse import Data

class Experiment(object):
  def __init__(self,name,paths):
    self.name = name
    self.data_paths = paths
    self._data = []
    self.__initialize_data()
  def __initialize_data(self):
    for data_path in self.data_paths:
      filename = os.path.basename(data_path)
      self._data.append(Data(filename,data_path))

  def append(self, value):
    self._data.append(value)

  def __setitem__(self, index, data):
      self._data[index] = data

  def __getitem__(self, index):
      return self._data[index]

  def __str__(self):
    str_experiment = """
    Name: {}
    paths : 
    {}
    data : 
    {} 
    """
    return str_experiment.format(self.name,self.data_paths, [str(data) for data in self._data])