import os
from .data import Data
from .subject import Subjects, Subject


class Experiment(object):
  """
  Esta clase define el experimento que se divide en Placentero/Neutro/Displacentero
  Agrupa los CSV.
  """  
  def __init__(self, name, paths, id):
    self.name = name
    self.id = id
    self.data_paths = paths
    self.subjects = Subjects()
    self.data = []
    self.__initialize_data__()

  def __initialize_data__(self):
    for data_path in self.data_paths:
      filename = os.path.basename(data_path)
      self.data.append(Data(filename, data_path, self))

  def get_subjects(self):
    return self.subjects.get_subjects()

  def __setitem__(self, index, data):
      self.data[index] = data

  def __getitem__(self, index):
      return self.data[index]

  def __str__(self):
    str_experiment = """
    Name: {}
    paths : 
    {}
    data : 
    {} 
    """
    return str_experiment.format(self.name,self.data_paths, [str(data) for data in self._data])