from .single_channel_interface import SingleChannelInterface
from itertools import permutations , combinations, combinations_with_replacement, product 
from random import choice, sample


class SameChannel(SingleChannelInterface):

    def calculate_dataset(self):
        self.dataset = self.__set_same_channel__()

    def __set_same_channel__(self):
        """
        Establece el dataset decidiendo si pertenece al mismo canal o no.
        """      
        dataset_pos = []
        dataset_neg = []    
        key = "same_channel"
        self.dataset_metadata[key] = {"len":0, "positive_count": 0, "negative_count" : 0}
        subjects = self.get_subjects()
        iter_max = 99999
        k = 0
        while (min([len(dataset_pos), len(dataset_neg)]) <= self.max_data//2):
            subject = choice(subjects)
            data_idx = choice(range(self.len_data))
            chn1 = choice(self.channels)
            chn2 = choice(self.channels)
            idx_data1, idx_data2 = self.get_data_from_subjets(subject, data_idx, chn1, chn2)
            pos, neg = self.set_data_from_same_process(idx_data1, idx_data2, chn1, chn2, subject, key, extra_data=self.get_extra_data(subject, chn1, chn2, data_idx))
            if len(dataset_pos) < self.max_data//2:
                dataset_pos += pos
            if len(dataset_neg) < self.max_data//2:
                dataset_neg += neg
            if k >= iter_max:
                break
            k +=1
        """
        for subject in subjects:
            for data_idx in range(self.len_data):
                for i in range(self.channels_iter):
                    chn1 = choice(self.channels)
                    chn2 = choice(self.channels)
                    idx_data1, idx_data2 = self.get_data_from_subjets(subject, data_idx, chn1, chn2)
                    pos, neg = self.set_data_from_same_process(idx_data1, idx_data2, chn1, chn2, subject, key, extra_data=self.get_extra_data(subject, chn1, chn2, data_idx))
                    if len(dataset_pos) < self.max_data//2:
                        dataset_pos += pos
                    if len(dataset_neg) < self.max_data//2:
                        dataset_neg += neg
                    if min([len(dataset_pos), len(dataset_neg)]) > self.max_data//2:
                        break
            n_subject += 1
        """
        if self.balance_dataset:
            dataset_pos, dataset_neg = self.get_balanced_dataset(dataset_pos, dataset_neg) 
        self.dataset_metadata[key]["positive_count"] = len(dataset_pos)
        self.dataset_metadata[key]["negative_count"] = len(dataset_neg)
        self.dataset_metadata[key]["len"] = self.dataset_metadata[key]["positive_count"] + self.dataset_metadata[key]["negative_count"]
        return dataset_pos + dataset_neg


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