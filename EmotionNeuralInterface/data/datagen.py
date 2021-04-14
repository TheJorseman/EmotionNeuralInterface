from itertools import permutations , combinations, combinations_with_replacement, product 
from random import sample, choice, seed 
from math import ceil

seed(27)

class DataGen(object):
  """
  Esta clase sirve para generar los datos de entrenamiento/prueba o validación. 
  """  
  def __init__(self,subjects, tokenizer, 
                    channels="all", combinate_subjects=False, combine_data=False, 
                    balance_dataset=True, data_chn_sampling=-1, channels_iter=3000,
                    targets_cod = {"positive": 1, "negative":0}):
    """
    Constructor

    Args:
        subjects (Subjects): Sujetos con los que se van a generar los datos
        tokenizer (Tokenizer): Clase donde se almacenan los ids de los segmentos que se obtienen de la señal.
        channels (str, optional): Canales con los que se quiere trabajar. Defaults to "all".
        combinate_subjects (bool, optional): Sirve para que en la generación de los datos se puedan combinar los sujetos. Defaults to False.
        combine_data (bool, optional): Sirve para combinar segmentos de distintos estimulos. Defaults to False.
        balance_dataset (bool, optional): Balancea el dataset. Defaults to True.
        data_chn_sampling (int, optional): Define cuantos canales se pueden elegir para hacer un sampling y no generar con todos los canales. Defaults to -1.
        channels_iter (int, optional): Cuantas iteraciones de canales random se pueden generar. Defaults to 3000.
        targets_cod (dict, optional): Codificación de los que son similares a los que no. Defaults to {"positive": 1, "negative":0}.
    """    
                    
    self.subjects = subjects
    self.tokenizer = tokenizer
    self.combinate_subjects = combinate_subjects
    self.combine_data =combine_data
    self.balance_dataset = balance_dataset
    self.targets_cod = targets_cod 
    if isinstance(channels,list):
      self.channels = channels
    else:
      self.channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    self.len_data = min([len(subject.get_clean_data()) for subject in subjects])
    self.consecutive_dataset = []
    self.same_channel_dataset = []
    self.same_subject_dataset = []
    self.channels_iter = channels_iter
    self.data_chn_sampling = data_chn_sampling
    self.dataset_metadata = {}
    
    self.same_channels = False
    self.not_same_channels = False
    
  def get_consecutive_dataset(self):
    if len(self.consecutive_dataset) > 0:
      return self.consecutive_dataset
    self.__set_consecutive__()
    return self.consecutive_dataset

  def get_same_channel_dataset(self):
    if len(self.same_channel_dataset) > 0:
      return self.same_channel_dataset
    self.__set_same_channel__()
    return self.same_channel_dataset

  def get_same_subject_dataset(self):
    if len(self.same_subject_dataset) > 0:
      return self.same_subject_dataset
    self.__set_same_subject__()
    return self.same_subject_dataset

  def get_subects(self):
    subjects_iter = self.subjects
    if self.combinate_subjects:
      subjects_iter = self.get_subjects_combinations()
    return subjects_iter


  def get_tiny_custom_channel_dataset(self, samples):
    """
    Este método define la creación de un dataset pequeño para hacer overfitting y saber si la red esta aprendiendo o que pasa.

    Args:
        samples (int): Número de datos que se quiere extraer (por ahora se cambio)

    Returns:
        list: Dataset
    """    
    dataset_pos = []
    dataset_neg = []
    len_channels = int((samples/(len(self.subjects)*self.len_data))/2)
    #len_channels = 14 if len_channels>14 else len_channels
    len_channels = 14
    for subject in self.subjects:
      for data_idx in range(3):
        same_channels = sample(self.channels,len_channels)
        self.same_channels = same_channels
        not_same_channels = sample(list(combinations(same_channels,2)),len_channels)
        self.not_same_channels = not_same_channels
        for i in range(len_channels):
          for j in range(50):
            idx_data1, _= self.get_data_from_subjets(subject, data_idx, same_channels[i], same_channels[i])
            dataset_pos.append({"input1": idx_data1[j], "input2": idx_data1[j+2], "output": self.targets_cod["positive"], "chn1" : self.channels.index(same_channels[i]), "chn2" : self.channels.index(same_channels[i]), "subject1" : subject.id, "subject2" : subject.id, "estimulo": data_idx})
            idx_data3, idx_data4 = self.get_data_from_subjets(subject, data_idx, not_same_channels[i][0], not_same_channels[i][1])
            dataset_neg.append({"input1": choice(idx_data3), "input2": choice(idx_data4), "output": self.targets_cod["negative"], "chn1" : self.channels.index(not_same_channels[i][0]), "chn2" : self.channels.index(not_same_channels[i][1]), "subject1" : subject.id, "subject2" : subject.id, "estimulo": data_idx})
    return dataset_pos + dataset_neg


  def get_tiny_custom_channel_dataset_test(self, samples):
    """
    Este método define la creación de un dataset pequeño para hacer overfitting y saber si la red esta aprendiendo o que pasa.
    Esta es otra versión con algunas mejoras. Itera sobre cada sujeto/tipo de dato/canal y extrae datos del mismo canal.
    Args:
        samples (int): Número de datos que se quiere extraer (por ahora se cambio)

    Returns:
        list: Dataset
    """    
    dataset_pos = []
    dataset_neg = []
    len_channels = int((samples/(len(self.subjects)*self.len_data))/2)
    #len_channels = 14 if len_channels>14 else len_channels
    len_channels = 3
    #len_channels = 10
    step = 2
    for subject in self.subjects:
      for data_idx in range(3):
        same_channels = self.same_channels if self.same_channels else sample(self.channels,len_channels)
        not_same_channels = self.not_same_channels if self.same_channels else sample(list(combinations(same_channels,2)),len_channels)
        for i in range(len_channels):
          idx_data1, _= self.get_data_from_subjets(subject, data_idx, same_channels[i], same_channels[i])
          for inx in range(len(idx_data1)-step):
            dataset_pos.append({"input1": idx_data1[inx], "input2": idx_data1[inx+step], "output": self.targets_cod["positive"], "chn1" : self.channels.index(same_channels[i]), "chn2" : self.channels.index(same_channels[i]), "subject1" : subject.id, "subject2" : subject.id, "estimulo": data_idx})
          idx_data3, idx_data4 = self.get_data_from_subjets(subject, data_idx, not_same_channels[i][0], not_same_channels[i][1])
          data_len = min([len(idx_data3), len(idx_data4)])
          # Hacer un producto
          for inx in range(data_len):
            dataset_neg.append({"input1": choice(idx_data3), "input2": choice(idx_data4), "output": self.targets_cod["negative"], "chn1" : self.channels.index(not_same_channels[i][0]), "chn2" : self.channels.index(not_same_channels[i][1]), "subject1" : subject.id, "subject2" : subject.id, "estimulo": data_idx})
    samp = min([len(dataset_pos), len(dataset_neg)])
    return sample(dataset_pos,samp) + sample(dataset_neg,samp)
    
  """
  Consecutive Dataset
  """  
  def __set_consecutive__(self):
    self.dataset_metadata["consecutive"] = {"len":0, "positive_count": 0, "negative_count" : 0}
    consecutive_dataset = []
    for subject in self.subjects:
      for data_ix in range(self.len_data):
        for channel in self.channels:
          index_splited_data = subject.get_indexs(data_ix, channel)
          consecutive_dataset += self.__consecutive_permutation_data__(index_splited_data, subject, channel, data_ix)
    self.consecutive_dataset = consecutive_dataset
    self.dataset_metadata["consecutive"]["len"] = len(self.consecutive_dataset) 

  def __consecutive_permutation_data__(self, idx_data, subject, channel, data_idx):
    """
    Almacena en un diccionario los valores de una permutación de dos listas de datos.
    Junto con la categoria a la que corresponden y otra información.
    Esto es para el dataset de tipo consecutivo.
    Args:
        idx_data (dato): Lista con los indices de los segmentos.
        subject (Subject): Sujeto al que pertenecen los datos.
        channel (str): Canal de los datos. 
        data_idx (int): Tipo de estimulo al que pertenece.
    Returns:
        list: Dataset
    """    
    cons_data_pos = []
    cons_data_neg = []
    perm_index = list(permutations(idx_data, 2)) 
    for i in range(len(perm_index)):
      max_indx = max(perm_index[i])
      if perm_index[i][1] - 1 == perm_index[i][0]:
        # Si es consecuente
        cons_data_pos.append({"input1": perm_index[i][0], "input2": perm_index[i][1], "output": self.targets_cod["positive"], "chn1" : channel, "chn2" : channel, "subject1" : subject.id, "subject2" : subject.id, "estimulo": data_idx})
      else:
        # No es consecuente
        cons_data_neg.append({"input1": perm_index[i][0], "input2": perm_index[i][1], "output": self.targets_cod["negative"] , "chn1" : channel, "chn2" : channel, "subject1" : subject.id, "subject2" : subject.id, "estimulo": data_idx})
    if self.balance_dataset:
      cons_data_pos, cons_data_neg = self.get_balanced_dataset(cons_data_pos, cons_data_neg)
    self.dataset_metadata["consecutive"]["positive_count"] += len(cons_data_pos)
    self.dataset_metadata["consecutive"]["negative_count"] += len(cons_data_neg)
    return cons_data_pos + cons_data_neg

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

  def __set_same_channel__(self):
    """
    Establece el dataset decidiendo si pertenece al mismo canal o no.
    """      
    dataset_pos = []
    dataset_neg = []    
    key = "same_channel"
    self.dataset_metadata[key] = {"len":0, "positive_count": 0, "negative_count" : 0}
    subjects = self.get_subects()
    for subject in subjects:
      for data_idx in range(self.len_data):
        for i in range(self.channels_iter):
          chn1 = choice(self.channels)
          chn2 = choice(self.channels)
          idx_data1, idx_data2 = self.get_data_from_subjets(subject, data_idx, chn1, chn2)
          pos, neg = self.set_data_from_same_process(idx_data1, idx_data2, chn1, chn2, subject, key, extra_data=self.get_extra_data(subject, chn1, chn2, data_idx))
          dataset_pos += pos
          dataset_neg += neg
    if self.balance_dataset:
      dataset_pos, dataset_neg = self.get_balanced_dataset(dataset_pos, dataset_neg) 
    self.same_channel_dataset = dataset_pos + dataset_neg
    self.dataset_metadata[key]["positive_count"] = len(dataset_pos)
    self.dataset_metadata[key]["negative_count"] = len(dataset_neg)
    self.dataset_metadata[key]["len"] = len(self.same_channel_dataset) 
    return


  def get_extra_data(self, subject, chn1, chn2, data_idx):
    """
    Agrega al dataset información extra como el sujeto o los canales a los que pertenecen los datos.
    Args:
        subject (Subject): Sujeto
        chn1 (str): Canal 1
        chn2 (str): Canal 2
        data_idx (int): Tipo de estimulo

    Returns:
        dict: diccionario actualizado
    """    
    extra_data = {"chn1": self.channels.index(chn1), "chn2": self.channels.index(chn2)}
    if self.combinate_subjects:
      extra_data.update({"subject1": subject[0].id, "subject2":subject[1].id, "estimulo": data_idx})
    else:
      extra_data.update({"subject1": subject.id, "subject2":subject.id, "estimulo": data_idx})
    return extra_data

  def set_data_from_same_process(self, data1, data2, var1, var2, subjects, key, extra_data={}):
    """
    Establece a partir de dos tipos de datos si se trata del mismo proceso (Si es el mismo canal o mismo sujeto)

    Args:
        data1 (list): Data 1 
        data2 (list): Data 2
        var1 (obj): Variable a evaluar 1 
        var2 (obj): Variable a evaluar 2 
        subjects (object): Sujetos o sujeto
        key ([type]): No se para que esta esto jajajaa
        extra_data (dict, optional): Datos extra a poner en el dataset. Defaults to {}.

    Returns:
        tuple: Tupla con los dos datasets (positivo y negativo)
    """    
    positive = []
    negative = []
    if self.data_chn_sampling == -1:
      idx_data1 = data1
      idx_data2 = data2
    else:
      idx_data1 = sample(data1,self.data_chn_sampling)
      idx_data2 = sample(data2,self.data_chn_sampling)
    product_dataset = list(product(idx_data1, idx_data2))
    for data in product_dataset:
      if var1 == var2:
        positive.append({"input1": data[0], "input2": data[1], "output": self.targets_cod["positive"]}.update(extra_data))
      else:
        negative.append({"input1": data[0], "input2": data[1], "output": self.targets_cod["negative"]}.update(extra_data))
    return positive,negative

  def get_data_from_subjets(self, subject, data_ix, chn1, chn2):
    """
    Esta función sirve principalmente para extraer los datos por ejemplo, cuando se trata de iterar sobre un sujeto o combinar 
    Los sujetos para extraer la información de distintos sujetos.
    Args:
        subject (obj): Si el objeto es una tupla entonces extrae información de cada sujeto de tipo Subject de otro modo es un objeto de tipo subject
        data_ix (int): tipo de dato
        chn1 (str): Canal 1
        chn2 (str): Canal 2

    Returns:
        tuple: Tupla con los datos
    """    
    if type(subject) == type(tuple()):
      idx_data1 = subject[0].get_indexs(data_ix, chn1)
      idx_data2 = subject[1].get_indexs(data_ix, chn2)
    else:
      idx_data1 = subject.get_indexs(data_ix, chn1)
      idx_data2 = subject.get_indexs(data_ix, chn2)
    return idx_data1, idx_data2

  def __set_same_subject__(self):
    """
    Crea el dataset con la condición si es el mismo sujeto o no.
    """    
    dataset_pos = []
    dataset_neg = []     
    key = "same_subject"
    self.dataset_metadata[key] = {"len":0, "positive_count": 0, "negative_count" : 0}
    subjects = self.get_subjects_combinations()
    for subject in subjects:
      for data_idx in range(self.len_data):
        for i in range(self.channels_iter):
          chn1 = choice(self.channels)
          chn2 = choice(self.channels)
          idx_data1, idx_data2 = self.get_data_from_subjets(subject, data_idx, chn1, chn2)
          pos, neg = self.set_data_from_same_process(idx_data1, idx_data2, subject[0].id, subject[1].id, subject, key, extra_data=self.get_extra_data(subject, chn1, chn2, data_idx))
          dataset_pos += pos
          dataset_neg += neg
    if self.balance_dataset:
      dataset_pos, dataset_neg = self.get_balanced_dataset(dataset_pos, dataset_neg)   
    self.same_subject_dataset = dataset_pos + dataset_neg
    self.dataset_metadata[key]["positive_count"] = len(dataset_pos)
    self.dataset_metadata[key]["negative_count"] = len(dataset_neg)
    self.dataset_metadata[key]["len"] = len(self.same_subject_dataset) 


  def get_subjects_combinations(self):
    """
    Extrae combinaciones de los sujetos

    Returns:
        list: Lista con la combinación de los mismos sujetos y de diferentes.
    """    
    subject_dataset = []
    subjects_comb = list(combinations_with_replacement(self.subjects,2))
    same_subjects = [sub_pair for sub_pair in subjects_comb if sub_pair[0].id == sub_pair[1].id]
    diff_subjects = [sub_pair for sub_pair in subjects_comb if sub_pair[0].id != sub_pair[1].id]
    diff_subjects_samp = sample(diff_subjects,len(same_subjects))
    return same_subjects + diff_subjects_samp 

