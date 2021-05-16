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
    if type(data["input1"]) == type(list()):
      return self.handle_multiple_channel(data)
    else:
      return self.handle_single_channel(data)


  def handle_multiple_channel(self, data):
    output = {}
    input1_ids = data["input1"]
    input1_matrix_list = [self.tokenizer.full_dataset[tid] for tid in input1_ids]
    output["input1"] = torch.tensor(input1_matrix_list, dtype=torch.float32)
    input2_ids = data["input2"]
    input2_matrix_list = [self.tokenizer.full_dataset[tid] for tid in input2_ids]
    output["input2"] = torch.tensor(input2_matrix_list, dtype=torch.float32)
    output["output"] = torch.tensor(data["output"], dtype=torch.int)
    output["channels"] = data["channels"]
    output["subjects"] = data["subjects"]
    output["stimulus"] = data["stimulus"]
    return output

  def handle_single_channel(self, data):
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