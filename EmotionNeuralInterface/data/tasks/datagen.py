from random import sample

class DataGenTask(object):
  """
  Esta clase sirve para generar los datos de entrenamiento/prueba o validación. 
  """  
  def __init__(self, subjects, 
                    tokenizer,
                    gen_data="all", 
                    channels="all",
                    balance_dataset=True, 
                    data_chn_sampling=-1,
                    multichannel_create=False,
                    patience=50,
                    max_data=5000000):
    """
    Constructor

    Args:
        subjects (Subjects): Sujetos con los que se van a generar los datos
        tokenizer (Tokenizer): Clase donde se almacenan los ids de los segmentos que se obtienen de la señal.
        channels (str, optional): Canales con los que se quiere trabajar. Defaults to "all".
        balance_dataset (bool, optional): Balancea el dataset. Defaults to True.
        data_chn_sampling (int, optional): Define cuantos canales se pueden elegir para hacer un sampling y no generar con todos los canales. Defaults to -1.
        channels_iter (int, optional): Cuantas iteraciones de canales random se pueden generar. Defaults to 3000.
        targets_cod (dict, optional): Codificación de los que son similares a los que no. Defaults to {"positive": 1, "negative":0}.
    """    
    self.gen_data = gen_data
    self.subjects = subjects
    self.tokenizer = tokenizer
    self.balance_dataset = balance_dataset
    self.max_data = max_data
    if isinstance(channels,list):
      self.channels = channels
    else:
      self.channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    self.len_data = min([len(subject.get_clean_data()) for subject in subjects])
    self.dataset = []
    self.data_chn_sampling = data_chn_sampling
    self.dataset_metadata = {}
    self.multichannel_create = multichannel_create
    self.patience = patience

  def get_dataset(self):
    if len(self.dataset) > 0:
      return self.resample_data(self.dataset)
    self.calculate_dataset()
    return self.resample_data(self.dataset)

  def resample_data(self, dataset):
    if len(dataset) > self.max_data:
      return sample(dataset, self.max_data)
    return dataset

  def calculate_dataset(self):
    return

  def get_subjects(self):
    return self.subjects


  def get_balanced_dataset(self, data1, data2):
    """
    Balancea el dataset

    Args:
        data1 (list): Dataset 1 
        data2 (list): Dataset 2 

    Returns:
        list: Dataset balanceado
    """    
    if len(data1)>len(data2):
      data1_new = sample(data1,len(data2))
      return data1_new, data2
    elif len(data1) < len(data2):
      data2_new = sample(data2,len(data1))
      return data1, data2_new
    return  data1, data2
