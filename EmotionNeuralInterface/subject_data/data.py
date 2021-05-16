from pandas import read_csv, DataFrame, Series
from datetime import datetime, timedelta
from sklearn import preprocessing
from numpy import array,arange,nan

class Data(object):
  """
  Esta clase es para leer un archivo CSV de los experimentos para posteriormente
  Extraer metadatos, datos validos y exportar a trainset.
  """  
  def __init__(self, name, path, experiment, experiments_time=75):
    self.name = name
    self.path = path
    self.csv = read_csv(path)
    self.experiment = experiment
    self.header = read_csv(path,nrows=0)
    self.data = read_csv(path, header=None,skiprows=1)
    self.__set_data__()
    self.experiment_time = experiments_time
    self.set_markers()
    # Estos canales son los definidos por el dispositivo con el que se hicieron los experimentos
    self.channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
  
  def get_subject_name(self):
    """
    Return the subject name
    """    
    return self.metadata["subject"]

  def __set_data__(self):
    """
    Se extraen los metadatos y los datos limpios.
    """    
    self.metadata = self.get_metadata_from_df(self.header)
    self.df_data = self.get_clean_dataframe(self.data, self.metadata)

  def parse_csv_datetime(self,value):
    """
    Parsea la fecha y hora de lo obtenido en el csv como metadato.
    Args:
        value (str): String que contiene la fecha.

    Returns:
        datetime: Fecha parseada.
    """    
    return datetime.strptime(value, '%y.%m.%d %H.%M.%S')

  def get_metadata_from_df(self,df):
    """
    Se extrane los metadatos del CSV.

    Args:
        df (dataframe): dataframe del archivo csv.

    Returns:
        dict: metadatos
    """    
    columns = list(df.columns.values)
    metadata = {}
    for column in columns:
      values = column.split(":")
      key,value = values[0].strip(),values[1].strip()
      metadata[key] = value
    # Parse recorded time
    key = "recorded"
    metadata[key] = self.parse_csv_datetime(metadata[key])
    return metadata

  def get_clean_dataframe(self, df, metadata, data_column="labels"):
    """
    Regresa un dataframe limpio con los valores por cada canal (columna)

    Args:
        df (dataframe): Dataframe del CSV.
        metadata (dict): Metadatos
        data_column (str, optional): Llave de los metadatos que contiene las columnas correspondientes. Defaults to "labels".

    Returns:
        [type]: [description]
    """    
    new_columns = metadata[data_column].split(" ")
    new_df = DataFrame(df.to_numpy(),columns=new_columns)
    return new_df

  def set_markers(self):
    """
    Se establecen los marcadores de inicio y fin del experimento.
    """    
    self.index_init_marker = self.df_data["MARKER"].loc[lambda x: x==1.0].index[0]
    self.index_end_marker = self.index_init_marker + self.experiment_time * int(self.metadata["sampling"])
    if self.index_end_marker > self.df_data.shape[0]:
      self.index_end_marker = self.df_data.shape[0]
    return

  def get_util_data(self, factor=1):
    """
    Regresa un dataframe con los canales como columnas y diversos datos utiles.

    Returns:
        dataframe: Dataframe que corresponde a los valores por cada canal y cuando se inicio el experimento
    """ 
    exp = self.df_data
    index_marker = exp['MARKER'].loc[lambda x: x==1.0].index[0]
    df = exp[index_marker:]
    new_df = DataFrame()
    #x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    for channel in self.channels:
      values = df[channel].values.reshape(-1, 1)
      x_scaled = min_max_scaler.fit_transform(values)
      new_df[channel] = Series(x_scaled.reshape(-1)) * factor
    #############
    return new_df
    #return DataFrame(x_scaled,columns=list(df.columns.values))*1

  def to_trainset(self):
    """
    Se convierte el CSV a un formato compatible con 
    https://trainset.geocene.com/
    Returns:
        dataframe: Dataframe Convertido
    """    
    new_df_trainset = DataFrame()
    data = self.df_data
    channels = self.channels
    new_df_trainset_data = {"series":[], "timestamp":[] ,"value":[]}
    #date_format = "%Y-%m-%dT%H:%M:%S.000Z"
    timestamp = self.metadata["recorded"]
    #time_inc = 1000/int(self.metadata["sampling"])
    time_inc = 1000
    for index in range(data.shape[0]):
      for channel in channels:
        new_df_trainset_data["series"].append(channel)
        new_df_trainset_data["value"].append(data[channel][index])
        new_df_trainset_data["timestamp"].append(timestamp.astimezone().replace(microsecond=0).isoformat())
        #new_df_trainset_data["label"].append(0)
      timestamp += timedelta(milliseconds=time_inc)
    df = new_df_trainset.from_dict(new_df_trainset_data)
    df["label"] = nan
    return df


  def __str__(self):
    unshowed_keys = ["labels"]
    str_data = ""
    for key,value in self.metadata.items():
      if key in unshowed_keys:
        continue
      str_data += "{} : {}".format(key,value) + "\r\n"
    return str_data
