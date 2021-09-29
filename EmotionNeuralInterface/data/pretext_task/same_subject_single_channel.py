from .single_channel_interface import SingleChannelInterface
from random import choice

class SameSubject(SingleChannelInterface):

    def calculate_dataset(self):
        self.dataset = self.__set_same_subject__()


    def __set_same_subject__(self):
        """
        Crea el dataset con la condición si es el mismo sujeto o no.
        """    
        dataset_pos = []
        dataset_neg = []     
        key = "same_subject"
        self.dataset_metadata[key] = {"len":0, "positive_count": 0, "negative_count" : 0}
        subjects = self.get_subjects_combinations()
        iter_max = 99999
        k = 0
        while (min([len(dataset_pos), len(dataset_neg)]) <= self.max_data//2):
            subject = choice(subjects)
            data_idx = choice(range(self.len_data))
            chn1 = choice(self.channels)
            chn2 = choice(self.channels)
            idx_data1, idx_data2 = self.get_data_from_subjets(subject, data_idx, chn1, chn2)
            pos, neg = self.set_data_from_same_process(idx_data1, idx_data2, subject[0].id, subject[1].id, subject, key, extra_data=self.get_extra_data(subject, chn1, chn2, data_idx))
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
                    pos, neg = self.set_data_from_same_process(idx_data1, idx_data2, subject[0].id, subject[1].id, subject, key, extra_data=self.get_extra_data(subject, chn1, chn2, data_idx))
                    if len(dataset_pos) < self.max_data//2:
                        dataset_pos += pos
                    if len(dataset_neg) < self.max_data//2:
                        dataset_neg += neg
                    if min([len(dataset_pos), len(dataset_neg)]) > self.max_data//2:
                        break
        """
        if self.balance_dataset:
            dataset_pos, dataset_neg = self.get_balanced_dataset(dataset_pos, dataset_neg)   
        self.dataset_metadata[key]["positive_count"] = len(dataset_pos)
        self.dataset_metadata[key]["negative_count"] = len(dataset_neg)
        self.dataset_metadata[key]["len"] = len(dataset_pos) + len(dataset_neg)
        return dataset_pos + dataset_neg