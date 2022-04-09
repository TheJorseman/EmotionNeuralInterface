import samplerate
from sklearn import preprocessing
from pandas import Series, DataFrame

class Reader(object):
    def __init__(self, path):
        self.path = path
        self.data = None
        self.metadata = None

    def resampling(self, new_sampling_rate):
        new_df = DataFrame()
        for i, column in enumerate(self.data.columns):
            new_df[column] = Series(samplerate.resample(self.data.iloc[:,i], new_sampling_rate/self.sampling_rate, 'sinc_best'))
        self.sampling_rate = new_sampling_rate
        self.data = new_df

    def normalize(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        for i in range(len(self.data.columns)):
            x_scaled = min_max_scaler.fit_transform(self.data.iloc[:,i].values.reshape(-1, 1))
            self.data.iloc[:,i] = Series(x_scaled.reshape(-1))