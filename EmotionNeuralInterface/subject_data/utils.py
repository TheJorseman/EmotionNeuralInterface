from EmotionNeuralInterface.subject_data.experiment import Experiment
from EmotionNeuralInterface.subject_data.subject import Subjects

def create_subject_data(experiments_paths):
    experiments = []
    subjects = Subjects()
    for folder,file_paths in experiments_paths.items():
        exp = Experiment(folder,file_paths, len(experiments))
        for data in exp.data:
            subjects.append(data, exp)
        experiments.append(exp) 
    return experiments,subjects