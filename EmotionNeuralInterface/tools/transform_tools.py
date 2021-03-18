from EmotionNeuralInterface.data.subject_data import Subjects

def experiment_to_subject(experiments):
  subjects = Subjects()
  for exp in experiments.values():
    for data in exp:
      subjects.append(data)
  return subjects