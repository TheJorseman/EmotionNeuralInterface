from .single_channel_interface import SingleChannelInterface

from itertools import permutations
from random import choice, sample

class Consecutive(SingleChannelInterface):

    def calculate_dataset(self):
        self.dataset = self.__set_consecutive__()

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
        self.dataset_metadata["consecutive"]["len"] = len(self.consecutive_dataset) 
        return consecutive_dataset

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