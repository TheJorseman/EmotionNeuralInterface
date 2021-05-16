from .datagen import DataGen

from itertools import permutations , combinations, combinations_with_replacement, product 
from random import sample, choice, seed, randint
from math import ceil, floor


class SingleChannelInterface(DataGen):


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