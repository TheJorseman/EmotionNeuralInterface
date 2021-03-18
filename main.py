from EmotionNeuralInterface.tools.paths_utils import get_paths_experiment
from EmotionNeuralInterface.tools.experiment import Experiment
from EmotionNeuralInterface.tools.transform_tools import experiment_to_subject
from EmotionNeuralInterface.data.tokenizer import Tokenizer
from EmotionNeuralInterface.data.datagen import DataGen
from EmotionNeuralInterface.data.dataset import NetworkDataSet
from EmotionNeuralInterface.tools.utils import split_data_by_len
from EmotionNeuralInterface.model.model import SiameseLinearNetwork
from EmotionNeuralInterface.model.loss import ContrastiveLoss
from EmotionNeuralInterface.model.model import SiameseNetwork

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
#from sklearn.model_selection import train_test_split
import os
from random import shuffle, sample, seed
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

tokenizer = Tokenizer(shuffle_subjets, window_size=1024, stride=256)
shuffle(shuffle_subjets)
channel_iters = 300
print("Se Cargaron los datos")
train_subjets, other_subjets = split_data_by_len(shuffle_subjets,14)
validation_subjets, test_subjets = split_data_by_len(other_subjets,3)
# Datasets
# get_consecutive_dataset
# get_same_channel_dataset
# get_same_subject_dataset
# Train Dataset
target_cod = {"positive": 1, "negative":0}
train_data_generator = DataGen(train_subjets, tokenizer, combinate_subjects=True, channels_iter=channel_iters, targets_cod=target_cod)
data_train = train_data_generator.get_same_channel_dataset()
print("Entrenamiento")
print(train_data_generator.dataset_metadata)
# Validation Dataset
validation_data_generator = DataGen(validation_subjets, tokenizer, combinate_subjects=True, channels_iter=channel_iters, targets_cod=target_cod)
data_validation = validation_data_generator.get_same_channel_dataset()
print("Validacion")
print(validation_data_generator.dataset_metadata)
# Test Dataset
test_data_generator = DataGen(test_subjets, tokenizer, combinate_subjects=True, channels_iter=channel_iters, targets_cod=target_cod)
data_test = test_data_generator.get_same_channel_dataset()
print("Test")
print(test_data_generator.dataset_metadata)
# Create the datasets
training_set = NetworkDataSet(data_train, tokenizer)
validation_set = NetworkDataSet(data_validation, tokenizer) 
testing_set = NetworkDataSet(data_test, tokenizer)

TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
TEST_BATCH_SIZE = 2
LEARNING_RATE = 1e-04

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

validation_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **validation_params)
testing_loader = DataLoader(testing_set, **test_params)

print("Se cargaron los datasets")

writer = SummaryWriter()

def calcuate_metric(outputs, targets):
  o_labels = torch.cuda.IntTensor([target_cod["positive"] if tensor.item() < margin else target_cod["negative"] for tensor in outputs])
  n_correct = (o_labels==targets).sum().item()
  return n_correct

def evaluate_model():
  n = 1
  n_correct = 0
  examples = 0
  for _,data in enumerate(validation_loader, 0):
    input1 = data["input1"].to(device)
    input2 = data["input2"].to(device)
    target = data["output"].to(device)
    output1, output2 = model(input1, input2)
    loss_contrastive = loss_function(output1, output2, target.unsqueeze(1))
    eucledian_distance = F.pairwise_distance(output1, output2)
    n_correct += calcuate_metric(eucledian_distance, target)
    examples += target.size(0)
    if n % 10000 == 0:
      break
    n += 1
  print ("Accuracy: ", n_correct/examples)
  writer.add_scalar("Accuracy/Validation", n_correct/examples, epoch)
  return

def train(epoch):
  n = 1
  n_correct = 0
  examples = 0
  for _,data in enumerate(training_loader, 0):
    input1 = data["input1"].to(device)
    input2 = data["input2"].to(device)
    target = data["output"].to(device)
    output1, output2 = model(input1, input2)
    loss_contrastive = loss_function(output1, output2, target.unsqueeze(1))
    loss_contrastive.backward()
    optimizer.step()
    print("Epoch {} Current loss {}\n".format(epoch,loss_contrastive.item()))
    writer.add_scalar("Loss/train", loss_contrastive.item(), epoch)
    eucledian_distance = F.pairwise_distance(output1, output2)
    n_correct += calcuate_metric(torch.round(eucledian_distance), target)
    examples += target.size(0)
    print(eucledian_distance)
    print(target)
    print("Acc ", (n_correct/examples)*100)
    writer.add_scalar("Accuracy/train", (n_correct/examples)*100, epoch)
    n += 1
  print("Validacion del modelo")
  evaluate_model()
  return

model_name = "model-cnn-128-64-32-16-same-chn-15-relu-p1"
folder = "models"
ext = ".pt"
full_path = os.path.join(folder, model_name+ext)

if model_name + ext in os.listdir(folder):
  print("Se cargo el modelo")
  model = torch.load(model_name)
else:
  #model = SiameseLinearNetwork((128,64,128,256))
  model = SiameseNetwork()
#model = SiameseNetwork()
model.to(device)
margin = 15
loss_function = ContrastiveLoss(margin)
#margin = 60/2
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

torch.set_grad_enabled(True)

EPOCHS=20
init_epoch = 0
print("Entrenamiento")
model.train()
for epoch in range(init_epoch, init_epoch + EPOCHS):
    train(epoch)
    torch.save(model, "{}/{}-epoch-{}{}".format(folder, model_name, epoch, ext))
    print("Model Saved Successfully")

torch.save(model, full_path)

print("Model Saved Successfully")
writer.flush()
writer.close()

def test(model):
  #model_load.eval()
  y_real = []
  y_predict = []
  y_distance = []
  n_correct = 0
  examples = 0
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
    #writer.add_scalar("Test/Acc", n_correct/examples, 0)
  return y_real, y_predict, y_distance

model.eval()
y_real, y_predict, y_distance = test(model)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandas import crosstab
import numpy as np  
#import pdb;pdb.set_trace()
Y_P = np.array(y_predict)
Y_V = np.array(y_real)
confusion_matrix = crosstab(Y_P, Y_V, rownames=['Real'], colnames=['Predicción'])
confusion_matrix

print(classification_report(Y_V, Y_P))

#import pdb;pdb.set_trace()
