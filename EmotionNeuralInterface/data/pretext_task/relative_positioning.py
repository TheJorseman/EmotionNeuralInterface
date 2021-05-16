from EmotionNeuralInterface.data.pretext_task.multiple_channel_interface import MultipleChannels
from math import floor
from random import randint

class RelativePositioning(MultipleChannels):

    def calculate_dataset(self):
        key = "relative_positioning_multiple_channel"
        self.dataset_metadata[key] = {"len":0, "positive_count": 0, "negative_count" : 0}
        self.dataset_pos = []
        self.dataset_neg = []
        i = 0
        while (len(self.dataset_pos) + len(self.dataset_neg)) < self.dataset_len or i <= self.max_num_iter:
            dataset_pos, dataset_neg = self.relative_positioning()
            dataset_pos, dataset_neg = self.balance_dataset_fn(dataset_pos, dataset_neg)
            self.dataset_pos += dataset_pos
            self.dataset_neg += dataset_neg
            i += 1
        self.update_metadata(key, self.dataset_pos, self.dataset_neg)
        self.dataset = self.dataset_pos + self.dataset_neg

    def balance_dataset_fn(self, dataset_pos, dataset_neg):
        if self.balance_dataset:
            return self.get_balanced_dataset(dataset_pos, dataset_neg)
        return dataset_pos, dataset_neg


    def relative_positioning(self):
        dataset_pos = []
        dataset_neg = []
        channels_split = self.split_channels()
        for subject in self.subjects:
            for data_idx in range(self.len_data):
                for chn_list in channels_split:
                    values, indexs = self.get_random_data_matrix(subject, data_idx, chn_list)
                    value_dict = self.in_same_context(values, indexs)
                    value_dict.update(self.get_extra_data(chn_list, subject, data_idx))
                    dataset_pos, dataset_neg = self.update_datasets(dataset_pos, dataset_neg, value_dict)
        return dataset_pos, dataset_neg

    def update_datasets(self, dataset_pos, dataset_neg, value_dict):
        if value_dict.get("output") == self.targets_cod["positive"]:
            dataset_pos.append(value_dict)
        else:
            dataset_neg.append(value_dict)
        return dataset_pos, dataset_neg

    def in_same_context(self, values, indexs):
        result = {  
                "input1": values[0] ,
                "input2": values[1]
                }
        if abs(indexs[0] - indexs[1]) <= self.t_pos_max:
            result.update({"output": self.targets_cod["positive"]})
        else:
            result.update({"output": self.targets_cod["negative"]})
        return result


    def get_extra_data(self, channel_list, subjects, data_idx):
        subjects = subjects if type(subjects) == type(list()) else [subjects]
        return {"channels" : [self.channels.index(chn) for chn in channel_list],
                "subjects" : [subject.id for subject in subjects],
                "stimulus" : [subject.data[data_idx].experiment.id for subject in subjects ]
                }
