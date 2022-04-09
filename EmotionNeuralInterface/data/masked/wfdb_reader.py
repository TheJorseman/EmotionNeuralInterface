from .reader import Reader
import wfdb
from pathlib import Path
from pandas import DataFrame
import numpy as np

class WFDBReader(Reader):
    def __init__(self, path, channels=['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']) -> None:
        super(WFDBReader, self).__init__(path)
        self.path = Path(path)
        self.name = self.path.name
        self.raw = wfdb.rdrecord(str(Path(self.path, self.name)))
        self.__set_metadata__()
        self.channels = channels
        self.channels_mask = np.array([channel in self.channels for channel in self.metadata.get('channels')])
        self.data = DataFrame(self.raw.p_signal[:,self.channels_mask], columns=self.channels)
        self.sampling_rate = int(self.metadata.get('sampling'))
        self.normalize()

    def __set_metadata__(self):
        self.metadata = {
            'record_name': self.raw.record_name,
            'sampling': self.raw.fs,
            'len': self.raw.sig_len,
            'units': self.raw.units,
            'channels': self.raw.sig_name,
        }