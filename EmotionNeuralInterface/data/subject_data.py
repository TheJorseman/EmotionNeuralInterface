
class Subject(object):
  """
  Clase que almacena la información de cada sujeto.
  """  
  def __init__(self, name , data=[], subject_id=0):
    self.name = name
    self.data = data
    self.indexs = [dict() for i in range(len(data))]
    self.id = subject_id

  def set_indexs(self, data_index, channel, values):
    """
    Guarda en un diccionario el tipo de dato que contiene y a su vez el canal al que pertenece.

    Args:
        data_index (int): índice en el que está el estímulo.
        channel (str): Canal
        values (list): Lista con los valores del canal.
    """    
    self.indexs[data_index][channel] = values

  def get_indexs(self, data_index, channel):
    return self.indexs[data_index][channel]

  def append(self, data):
    self.data.append(data)
    self.indexs.append(dict())

  def get_clean_data(self):
    return [data.get_util_data() for data in self.data]

  def __getitem__(self, index):
    return self.data[index]


class Subjects(object):
  """
  Clase que alacena todos los sujetos.
  """  
  def __init__(self):
    self.data_by_subject = {}
    self.full_data = []

  def append(self, data):
    subject_name = data.metadata["subject"].lower()
    if not subject_name in self.data_by_subject:
      self.data_by_subject[subject_name] = Subject(subject_name, [data], subject_id=len(self.data_by_subject.keys()))
    else:
      self.data_by_subject[subject_name].append(data)

  def __setitem__(self, index, data):
    list(self.data_by_subject.values())[index] = data

  def __getitem__(self, index):
    return list(self.data_by_subject.values())[index]

  def __len__(self):
    return len(list(self.data_by_subject.values()))

  def copy_list(self):
    return list(self.data_by_subject.values()).copy()