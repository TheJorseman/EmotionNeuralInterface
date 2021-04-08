import torch
from torch.utils.data import Dataset

class NetworkDataSet(Dataset):
  """
  Esta clase hereda de la clase Dataset de Pytorch, en esta se generan en forma de tensor todos los datos para ser procesados por la red neuronal.
  """  
  def __init__(self, data, tokenizer):
    self.tokenizer = tokenizer
    self.data = data

  def __getitem__(self, i):
    data = self.data[i]
    output = {}
    input1_id = data["input1"]
    output["input1"] = torch.tensor(self.tokenizer.full_dataset[input1_id], dtype=torch.float32)
    input2_id = data["input2"]
    output["input2"] = torch.tensor(self.tokenizer.full_dataset[input2_id], dtype=torch.float32)
    output["output"] = torch.tensor(data["output"], dtype=torch.int)
    output["subject1"] = data["subject1"]
    output["chn1"] = data["chn1"]
    output["subject2"] = data["subject2"]
    output["chn2"] = data["chn2"]
    output["estimulo"] = data["estimulo"]
    return output

  def __len__(self):
    return len(self.data)