from EmotionNeuralInterface.tools.paths_utils import get_paths_experiment
from EmotionNeuralInterface.tools.experiment import Experiment
from EmotionNeuralInterface.tools.transform_tools import experiment_to_subject
from EmotionNeuralInterface.data.tokenizer import Tokenizer
from EmotionNeuralInterface.data.datagen import DataGen
from EmotionNeuralInterface.data.dataset import NetworkDataSet
from EmotionNeuralInterface.tools.utils import split_data_by_len
from EmotionNeuralInterface.model.model import SiameseLinearNetwork
from EmotionNeuralInterface.model.loss import ContrastiveLoss

import torch
from torch import cuda
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
#from sklearn.model_selection import train_test_split
import os
from random import shuffle, sample, seed

from sklearn.decomposition import PCA
from pandas import DataFrame
seed(27)


device = 'cuda' if cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)

exp_path = "C:/Users/migue/Documents/GitHub/Datasets/Experimento"
#exp_path = os.path.join("Users","migue","Documents","GitHub","Datasets","Experimento")  
experiments_paths = get_paths_experiment(exp_path)

experiments = {}
for folder,file_paths in experiments_paths.items():
  experiments[folder] = Experiment(folder,file_paths)

subjects = experiment_to_subject(experiments)

shuffle_subjets = subjects.copy_list()


shuffle(shuffle_subjets)
print("Se Cargaron los datos")
train_subjets, other_subjets = split_data_by_len(shuffle_subjets,14)
validation_subjets, test_subjets = split_data_by_len(other_subjets,3)

tokenizer = Tokenizer(test_subjets, window_size=1024, stride=256)
# Datasets
# get_consecutive_dataset
# get_same_channel_dataset
# get_same_subject_dataset

# Train Dataset
#train_data_generator = DataGen(train_subjets, tokenizer)
#data_train = train_data_generator.get_same_channel_dataset()
# Validation Dataset
#validation_data_generator = DataGen(validation_subjets, tokenizer)
#data_validation = validation_data_generator.get_same_channel_dataset()
# Test Dataset
targets_cod = {"positive": 0, "negative":1}
test_data_generator = DataGen(test_subjets, tokenizer, combinate_subjects=True, channels_iter=100, targets_cod=targets_cod)
data_test = test_data_generator.get_same_channel_dataset()
print(test_data_generator.dataset_metadata)
# Create the datasets
#training_set = NetworkDataSet(data_train, tokenizer)
#validation_set = NetworkDataSet(data_validation, tokenizer) 
testing_set = NetworkDataSet(data_test, tokenizer)

TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
LEARNING_RATE = 1e-04

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

validation_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

#training_loader = DataLoader(training_set, **train_params)
#validation_loader = DataLoader(validation_set, **validation_params)
testing_loader = DataLoader(testing_set, **test_params)
print("Se cargaron los datasets")

def calcuate_metric(outputs, targets):
  o_labels = torch.cuda.IntTensor([1 if tensor.item() < margin else 0 for tensor in outputs])
  n_correct = (o_labels==targets).sum().item()
  return n_correct


model_name = "model-cnn-128-64-32-16-same-chn-15-relu-p0.pt"
if model_name in os.listdir("models"):
  print("Se cargo el modelo")
  model = torch.load("models/"+model_name)
else:
    raise Warning("No se encuentra el modelo :(")
margin = 15
loss_function = ContrastiveLoss(margin)

def test(model):
  #model_load.eval()
  y_real = []
  y_predict = []
  y_distance = []
  n_correct = 0
  examples = 0
  df = list()
  datalen = len(testing_set)
  i = 0
  for _,data in enumerate(testing_loader, 0):
    input1 = data["input1"].to(device)
    input2 = data["input2"].to(device)
    targets = data["output"].to(device)
    output1, output2 = model(input1, input2)
    loss_contrastive = loss_function(output1, output2, targets.unsqueeze(1))
    eucledian_distance = F.pairwise_distance(output1, output2)
    y_real += [target.item() for target in targets]
    y_predict += [1 if tensor.item() < margin else 0 for tensor in eucledian_distance]
    n_correct += calcuate_metric(eucledian_distance, targets)
    examples += targets.size(0)
    print("Acc= ", n_correct/examples)
    #import pdb;pdb.set_trace()
    df.append({'Vector': output1.to("cpu").detach().numpy()[0], "Categ": data["output"].item(), "subject": data["subject"].item()})
    df.append({'Vector': output2.to("cpu").detach().numpy()[0], "Categ": data["output"].item(), "subject": data["subject"].item()})
    if i == 10000:
      break
    print("Completado: ", i)
    i += 1
    #writer.add_scalar("Test/Acc", n_correct/examples, 0)
  
  return y_real, y_predict, y_distance,DataFrame.from_dict(df)

model.eval()
y_real, y_predict, y_distance,df = test(model)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandas import crosstab
import numpy as np  

Y_P = np.array(y_predict)
Y_V = np.array(y_real)
confusion_matrix = crosstab(Y_P, Y_V, rownames=['Real'], colnames=['Predicción'])
confusion_matrix

print(classification_report(Y_V, Y_P))
import torch.nn as nn

#layers = [module for module in model.modules() if type(module) == nn.Linear]
#last_layer = layers[-1].weight


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


m = TSNE(n_components=3,learning_rate=50)
tsne_features = m.fit_transform(np.array(df.Vector.tolist()))

df["x"] = tsne_features[:,0]
df["y"] = tsne_features[:,1]
df["z"] = tsne_features[:,2]

sns.scatterplot(x="x",y="y", hue="Categ", data=df)
plt.show()
fig = px.scatter_3d(df, x='x', y='y', z='z',color=df.Categ)

fig_2d = px.scatter(df, x='x', y='y',color=df.Categ)

fig_2d.show()
fig.show()

"""
from umap import UMAP
umap_2d = UMAP(n_components=2, init='random', random_state=0)
umap_3d = UMAP(n_components=3, init='random', random_state=0)

proj_2d = umap_2d.fit_transform(np.array(df.Vector.tolist()))
proj_3d = umap_3d.fit_transform(np.array(df.Vector.tolist()))

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df.Categ
)
fig_3d = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=df.Categ)
    
fig_3d.update_traces(marker_size=5)

fig_2d.show()
fig_3d.show()

import pdb;pdb.set_trace()
"""


#