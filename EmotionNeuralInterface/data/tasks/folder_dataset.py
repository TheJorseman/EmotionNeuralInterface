from .datagen import DataGenTask
from itertools import permutations , combinations, combinations_with_replacement, product 
from random import choice, sample, randint
from numpy import ndarray

class FolderDataset(DataGenTask):

    def calculate_dataset(self):
        self.dataset = self.__set_folder_dataset__()

    def stop_condition(self, dataset):
        #return len(dataset) >= self.max_data
        
        return min([len(dataset[k]) for k in dataset.keys()]) <= self.max_data//self.len_data

    def __set_folder_dataset__(self):
        """
        Establece el dataset decidiendo su etiqueta dependiendo de su categoria en el directorio.
        """
        self.gen_data

        key = "finetuning_dataset"
        self.dataset_metadata[key] = {"len":0}
        subjects = self.get_subjects()
        if self.gen_data.lower() == "all":
            dataset = self.get_all_data(subjects)
            print(len(dataset))
            self.dataset_metadata[key] = {"len":len(dataset)}
            return dataset
        else:
            return self.valued_dataset(subjects)

    def valued_dataset(self, subjects):
        key = "finetuning_dataset"
        idx_values = []
        dataset = {k:[] for k in range(self.len_data)}
        iter_max = 99999
        k = 0
        data_idx_range = range(self.len_data)
        while (self.stop_condition(dataset)):
            subject = choice(subjects)
            data_idx = choice(data_idx_range)
            if len(dataset[data_idx]) > self.max_data//self.len_data:
                if k >= iter_max:
                    break
                k +=1
                continue
            chns = self.channels if self.multichannel_create else [choice(self.channels)]
            data, idx_values = self.get_data(subject, data_idx, chns, idx_values=idx_values)
            if not data:
                break
            dataset[data_idx].append(data)
        if self.balance_dataset:
            dataset = self.get_balanced_dataset(dataset)
        self.dataset_metadata[key] = {k: len(v) for k,v in dataset.items()}
        full_dataset = []
        for k,v in dataset.items():
            full_dataset += list(v)
        return full_dataset

    def get_chn_data(self, subject, data_ix, idx):
        if self.multichannel_create:
            matrix = []
            for chn in self.channels:
                id = subject.get_indexs(data_ix, chn)[idx]
                matrix.append(id)
            return [self.get_dict_data(matrix, data_ix, subject, self.channels)]
        all_data = []
        for chn in self.channels:
            id = subject.get_indexs(data_ix, chn)[idx]
            all_data.append(self.get_dict_data(id, data_ix, subject, chn))
        return all_data


    def get_all_data(self, subjects):
        full_dataset = []
        for subject in subjects:
            for data_ix in range(self.len_data):
                max_idx = len(subject.get_indexs(data_ix, self.channels[0])) - 1 
                for idx in range(max_idx):
                    full_dataset += self.get_chn_data(subject, data_ix, idx)
        return full_dataset

    def get_balanced_dataset(self, dataset):
        min_len = min([len(dataset[k]) for k in dataset.keys()])
        balanced_dataset = {k:[] for k in dataset.keys()}
        for k,v in dataset.items():
            balanced_dataset[k] = sample(v, min_len)
        return balanced_dataset

    def get_data(self, subject, data_ix, channels, idx_values=[], tries=50):
        for patience in range(self.patience):
            data, idx_values = self.get_tensor_data(subject, data_ix, channels, idx_values=idx_values, tries=tries)
            if data:
                data = data if self.multichannel_create else data[0]
                return self.get_dict_data(data, data_ix, subject, channels), idx_values
        return False, idx_values
            
    def get_dict_data(self, data, data_ix, subject, chns):
        return {
            'input' : data,
            'output': data_ix,
            'subject': subject.id,
            'channels': chns
        }

    def get_tensor_data(self, subject, data_ix, channels, idx_values=[], tries=50):
        max_idx = len(subject.get_indexs(data_ix, channels[0])) - 1 
        output = False
        chn = -1 if self.multichannel_create else channels[0]
        for i in range(tries):
            repeated = False
            idx_data = randint(0, max_idx)
            data = []
            for chn in channels:
                id = subject.get_indexs(data_ix, chn)[idx_data]
                if idx_data in idx_values:
                    repeated = True
                    break
                data.append(id)
                idx_values.append(id)
            if repeated:
                continue
            output = data
            break
        return output, idx_values