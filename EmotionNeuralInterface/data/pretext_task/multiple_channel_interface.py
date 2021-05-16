from EmotionNeuralInterface.data.pretext_task.datagen import DataGen
from math import floor
from random import randint

class MultipleChannels(DataGen):
    def __init__(self, subjects, 
                    tokenizer, 
                    channels='all', 
                    # Multiple Channel Config
                    multiple_channel_len=14, 
                    t_pos_max=6, 
                    dataset_len=500000, 
                    max_num_iter=300,
                    ########################
                    combinate_subjects=False, 
                    combine_data=False, 
                    balance_dataset=True, 
                    data_chn_sampling=-1, 
                    channels_iter=3000, 
                    targets_cod={'positive':1,'negative':0}):
        super(MultipleChannels,self).__init__(subjects, 
                                            tokenizer, 
                                            channels=channels, 
                                            combinate_subjects=combinate_subjects, 
                                            combine_data=combine_data, 
                                            balance_dataset=balance_dataset, 
                                            data_chn_sampling=data_chn_sampling, 
                                            channels_iter=channels_iter, 
                                            targets_cod=targets_cod)
        self.multiple_channel_len = multiple_channel_len
        self.t_pos_max = t_pos_max
        self.validate_multiple_channels()
        self.dataset_len = dataset_len
        self.max_num_iter = max_num_iter

    def validate_multiple_channels(self):
        if len(self.channels) < self.multiple_channel_len:
            raise Warning("Value of multiple_channel_len is more than disponible channels (14)")

    def split_channels(self):
        channels_list = []
        num_iter = floor(len(self.channels)/self.multiple_channel_len)
        for i in range(num_iter):
            ini_val = i*self.multiple_channel_len
            channels_list.append(self.channels[ini_val:ini_val+self.multiple_channel_len])
        return channels_list

    def get_random_data_matrix(self, subject, data_ix, channels, values=2):
        max_idx = len(subject.get_indexs(data_ix, channels[0])) - 1 
        output = []
        idx_values = []
        for i in range(values):
            data = []
            idx_values.append(randint(0,max_idx))
            for chn in channels:
                data.append(subject.get_indexs(data_ix, chn)[idx_values[-1]])
            output.append(data)
        return output, idx_values

    def update_metadata(self, key, positive, negative):
        self.dataset_metadata[key]["positive_count"] = len(positive)
        self.dataset_metadata[key]["negative_count"] = len(negative)
        self.dataset_metadata[key]["len"] = len(positive) +  len(negative)
        

    