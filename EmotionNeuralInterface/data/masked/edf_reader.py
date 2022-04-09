import os
import mne
from .reader import Reader


class EDFReader(Reader):
    def __init__(self, path) -> None:
        super(EDFReader, self).__init__(path)
        self.path = path
        self.raw = mne.io.read_raw_edf(path, preload=True)
        self.metadata = dict(self.raw.info)
        self.df = self.raw.to_data_frame()
        self.sampling_rate = int(self.metadata.get('sfreq'))
        self.data = self.clean_df()
        self.normalize()

    def clean_df(self):
        if 'Channel' in self.df.columns:
            self.df = self.df.drop(['Channel'], axis=1)
        return self.df.drop(['time'], axis=1)
 
 
