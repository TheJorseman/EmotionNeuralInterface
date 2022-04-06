from EmotionNeuralInterface.data.masked.csv_reader import CSVReader
from EmotionNeuralInterface.data.masked.edf_reader import EDFReader
from EmotionNeuralInterface.data.masked.wfdb_reader import WFDBReader
from pathlib import Path

import wfdb

def test_open_csv(dataset_path):
    for csv in Path(dataset_path).glob("*/*.csv"):
        CSVReader(csv)

def test_open_edf(dataset_path, search_filter):
    for edf in Path(dataset_path).glob(search_filter):
        edf = EDFReader(edf)
        #print(edf.sampling_rate)
        #print(edf.data)
        #edf.resampling(128)

def test_open_wfdb(dataset_path):
    folders = filter(lambda f: f.is_dir(), Path(dataset_path).glob("*"))
    for folder in folders:
        wfdb = WFDBReader(folder)
        #print(wfdb.sampling_rate)
        #print(wfdb.data)
        #wfdb.resampling(128)

def test_rsvp_task():
    dataset = "../Datasets/eeg-signals-from-an-rsvp-task-1.0.0"
    test_open_edf(dataset, "*/*.edf")

def test_movement_task():
    dataset = "../Datasets/eeg-motor-movementimagery-dataset-1.0.0"
    test_open_edf(dataset, "*/*/*.edf")

def test_mental_task():
    dataset = "../Datasets/eeg-during-mental-arithmetic-tasks-1.0.0"
    test_open_edf(dataset, "*.edf")


def test_cinc_challenge():
    dataset = r'G:\Downloads\Torrents\training'
    test_open_wfdb(dataset)

#path = '../Datasets/training/tr03-0005/tr03-0005'

#test_open_csv("data")
#test_rsvp_task()
#test_movement_task()
#test_mental_task()
test_cinc_challenge()
import pdb;pdb.set_trace()

