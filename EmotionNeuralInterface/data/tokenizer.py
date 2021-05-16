import numpy as np

class Tokenizer(object):
  """
  Esta clase sirve para "tokenizar" las series de tiempo, las convierte a ids y estos ids sirven para generar los datos y cuando se procesen
  obtener los datos.
  """  
  def __init__(self, subjects, 
                    window_size=128,
                    channels="all", 
                    pad_array=False, 
                    stride=64):
    """
    Constructor.
    Args:
        subjects (Subjects): lista con todos los sujetos.
        window_size (int, optional): El tamaño de la ventana en la que se quiere dividir. Defaults to 128.
        channels (str, optional): Canales con los que se quiere trabajar. Defaults to "all".
        pad_array (bool, optional): Sirve para que . Defaults to False.
        stride (int, optional): Desplazamiento sobre la señal cuando se "tokeniza". Defaults to 128.
    """    
    self.subjects = subjects
    self.window_size = window_size
    self.stride = stride
    self.pad_array = pad_array
    if isinstance(channels,list):
      self.channels = channels
    else:
      self.channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    self.full_dataset = []
    #self.full_dataset = []
    #self.convert_to_ids()
    self.stride_dataset()

  def stride_dataset(self):
    """
    Genera el split de la señal.
    """    
    for subject in self.subjects:
      for data_ix in range(len(subject.get_clean_data())):
        data = subject.get_clean_data()
        for channel in self.channels:
          splited_data = self.split_data(self.window_size, self.stride, data[data_ix][channel].values)
          r_i = len(self.full_dataset) 
          r_e = len(splited_data) 
          subject.set_indexs(data_ix, channel, list(range(r_i, r_i + r_e)))
          self.full_dataset += splited_data
    return

  def split_data(self, kernel, stride, data):
    """
    Corta la señal en pedazos y les asigna un id.

    Args:
        kernel (int): Window Size
        stride (int): Desplazamiento
        data (array): Array que se quiere dividir.

    Returns:
        list: Array dividido
    """    
    n = int((len(data)-kernel)/stride) + 1
    data_tok = []
    for i in range(n):
      data_tok.append(data[stride*i:stride*i + kernel])
    return data_tok


  def convert_to_ids(self):
    """
    Realiza todo el proceso de split
    """    
    for subject in self.subjects:
      for data_ix in range(len(subject.get_clean_data())):
        data = subject.get_clean_data()
        for channel in self.channels:
          splited_data = self.split_by_window(data[data_ix][channel].values)
          r_i = len(self.full_dataset) 
          r_e = len(splited_data) 
          subject.set_indexs(data_ix, channel, list(range(r_i, r_i + r_e)))
          self.full_dataset += splited_data

  def split_data_by_len(self, data, n):
      for i in range(0, len(data), n):  
          yield data[i:i + n] 

  def split_by_window(self, data):
    splited_data = list(self.split_data_by_len(data, self.window_size))
    if len(splited_data[-1]) == self.window_size:
      return splited_data
    if self.pad_array:
      splited_data[-1] = np.pad(splited_data[-1], self.window_size , mode="constant") 
      return splited_data
    return splited_data[:-1]