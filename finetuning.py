import json
from EmotionNeuralInterface.tools.paths_utils import get_paths_experiment, get_path
from EmotionNeuralInterface.subject_data.utils import create_subject_data
from EmotionNeuralInterface.data.tokenizer import Tokenizer


# Datasets Generators
from EmotionNeuralInterface.data.tasks.folder_dataset import FolderDataset
# Tools
from EmotionNeuralInterface.data.dataset import NetworkDataSet
from EmotionNeuralInterface.tools.utils import split_data_by_len
# Loss
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
# Classificator
from EmotionNeuralInterface.model.classificator import Classificator

from sklearn.neighbors import KNeighborsClassifier


import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
#from sklearn.model_selection import train_test_split
from pandas import DataFrame
import os
import pickle
from random import shuffle, sample, seed
import yaml

from datetime import datetime

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from pandas import crosstab
import numpy as np  
### PLOTS
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from umap import UMAP
#seed(27)
import logging
from pathlib import Path
import psutil

MEMORY_LIMIT = 95


logging.basicConfig(
    format = '%(asctime)-5s %(name)-15s %(levelname)-8s %(message)s',
    level  = logging.INFO,      
    filename = "emotional_neural_interface_logs_info.log", 
    filemode = "a"                      
)

class FinetuneModel(object):
    def __init__(self, config_file):
        self.data = self._get_dict_model_file(config_file)
        print(self.data)
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        torch.set_grad_enabled(False)
        self.model_load_config = {}
        if self.data['use_pretrained']:
            self.load_pretrained_model()
            self.set_pretrained_data()
        else:
            self.pretrained_model = False
            self.set_multichannel_create_from_file()
        self.load_dataset()
        self.load_tokenizer()
        self.dataset_subjects()
        self.datagen_config()
        self.load_dataset_batch()
        self.train_accuracy = 0
        self.dA_train = 0
        self.evaluation_accuracy = 0
        self.dA_eval = 0
    
    def _get_dict_model_file(self, config_file):
        if isinstance(config_file, str):
            with open(config_file) as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        elif isinstance(config_file, dict):
            return config_file
        else:
            raise Warning("Type of file not valid")


    def set_multichannel_create_from_file(self):
        clasificator = self._get_dict_model_file(self.data['model']['model_config_path'])
        self.multichannel_create = ('stagenet' in clasificator['use_model'] or 'stagenet' in clasificator['use_model'])

    def load_pretrained_model(self):
        path = self.data['load_model']['path']
        self.pretrained_model = torch.load(path)
        metadata = open(self.data['load_model']['metadata'], "r")
        metadata = json.load(metadata)
        name = os.path.basename(path)
        if name not in metadata.keys():
            self.pretrained_model_config = metadata
            self.set_multichannel_create_from_file()
        else:
            self.pretrained_model_config = metadata[name]
            self.pretrained_model_config['name'] = name
            self.multichannel_create = ('conv2d' in name)
        return

    def set_pretrained_data(self):
        self.data['tokenizer']['window_size'] = int(self.pretrained_model_config['window_size'])
        print("Window Size", self.data['tokenizer']['window_size'])
        self.data['tokenizer']['stride'] = int(self.pretrained_model_config['stride'])
        print("Stride", self.data['tokenizer']['stride'])
        return

    def load_dataset(self):
        path = get_path(self.data["dataset"]["dataset_path"])
        experiments_paths = get_paths_experiment(path)
        self.experiments, self.subjects = create_subject_data(experiments_paths)
        
    def load_tokenizer(self):
        data = self.data["tokenizer"]
        self.tokenizer = Tokenizer(self.subjects, window_size=data["window_size"], stride=data["stride"])

    def dataset_subjects(self):
        data = self.data["dataset_subjects"]
        self.use_validation = bool(data["validation_subjects"])
        self.train_subjets, other_subjets = split_data_by_len(self.subjects.copy_list(), data["train_subjects"])
        # Use validation subjects
        self.test_subjets = other_subjets
        if self.use_validation:
            self.validation_subjets, self.test_subjets = split_data_by_len(other_subjets, data["test_subjects"])
    
    def datagen_config(self):
        data = self.data["datagen_config"]
        #self.target_cod = data["target_codification"]
        #self.combinate_subjects = data["combinate_subjects"]
        #self.channel_iters = data["channel_iters"]
        # Dataset Len
        self.dataset_train_len = data['dataset_train_len']
        self.dataset_test_len = data['dataset_test_len']
        self.dataset_validation_len = data['dataset_validation_len']
        self.set_train_datagen()
        if self.use_validation:
            self.set_validation_datagen()
        self.set_test_datagen()

    def set_train_datagen(self):
        self.train_data_generator = self.get_dataset_generator(self.train_subjets, dataset_max_len=self.dataset_train_len)
        self.class_dim = self.train_data_generator.len_data
        self.data_train = self.train_data_generator.get_dataset()
        print("Entrenamiento")
        print(self.train_data_generator.dataset_metadata)
        self.training_set = NetworkDataSet(self.data_train, self.tokenizer)


    def set_validation_datagen(self):
        self.validation_data_generator = self.get_dataset_generator(self.validation_subjets, dataset_max_len=self.dataset_validation_len)
        self.data_validation = self.validation_data_generator.get_dataset()
        print("Validacion")
        print(self.validation_data_generator.dataset_metadata)
        self.validation_set = NetworkDataSet(self.data_validation, self.tokenizer)

    def set_test_datagen(self):
        self.test_data_generator = self.get_dataset_generator(self.test_subjets, dataset_max_len=self.dataset_test_len)
        self.data_test = self.test_data_generator.get_dataset()
        print("Test")
        print(self.test_data_generator.dataset_metadata)
        self.testing_set = NetworkDataSet(self.data_test, self.tokenizer)


    def get_dataset_generator(self, subjects, dataset_max_len=500000):
        multiple_channel_dict = self.data["datagen_config"]["multiple_channel"]
        if self.data["datagen_config"]["dataset"] == "folder_dataset":
            return FolderDataset(subjects, self.tokenizer, max_data=dataset_max_len, gen_data=self.data["datagen_config"]["gen_data"], multichannel_create=self.multichannel_create)      
        raise Warning("No valid Dataset")  


    def datagen_overfitting(self):
        train_data_generator = self.get_dataset(self.train_subjets)
        self.data_train = train_data_generator.get_tiny_custom_channel_dataset_test(self.data["datagen_config"]["train_dataset_len"])
        self.data_test = train_data_generator.get_tiny_custom_channel_dataset_test(self.data["datagen_config"]["train_dataset_len"])
        self.training_set = NetworkDataSet(self.data_train, self.tokenizer)
        self.validation_set = NetworkDataSet(self.data_train, self.tokenizer)
        self.testing_set = NetworkDataSet(self.data_test, self.tokenizer)
        self.print_dataset()

    def print_dataset(self):
        print("Train len: {}".format(len(self.data_train)))
        print("Test len: {}".format(len(self.data_test)))


    def load_dataset_batch(self):
        data = self.data["dataset_batch"]
        self.TRAIN_BATCH_SIZE = data["train_batch_size"]
        self.VALID_BATCH_SIZE = data["validation_batch_size"]
        self.TEST_BATCH_SIZE = data["test_batch_size"]
        #self.LEARNING_RATE = data["learning_rate"]
        ###### Train #######
        train_params = {'batch_size': self.TRAIN_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 0
                        }
        self.training_loader = DataLoader(self.training_set, **train_params)
        ###### Validation #######   
        if self.use_validation:
            validation_params = {'batch_size': self.VALID_BATCH_SIZE,
                                'shuffle': True,
                                'num_workers': 0
                            }
            self.validation_loader = DataLoader(self.validation_set, **validation_params)
        ###### Test #######
        test_params = {'batch_size': self.TEST_BATCH_SIZE,
                            'shuffle': True,
                            'num_workers': 0
                        }
        self.testing_loader = DataLoader(self.testing_set, **test_params)
     
    def get_model_features(self):
        return self.model_config['name']

    def get_model_name(self):
        return "model-{}-loss{}-epoch{}".format(self.get_model_features(), 
            self.data["loss"]["loss_function"], self.data["train"]["epochs"])

    def set_model_config(self):
        config_file = self.data["model"]["model_config_path"]
        if isinstance(config_file, str):
            with open(config_file) as f:
                self.model_config = yaml.load(f, Loader=yaml.FullLoader)
                print(self.model_config)
        elif isinstance(config_file, dict):
            self.model_config = config_file
        else:
            raise Warning("Type of file not valid")

    def get_model_path(self, path):
        if path == 'last':
            path = self.data['model']['folder_save']
            folder = sorted(Path(path).iterdir(), key=os.path.getmtime)[-2]
            model = sorted(Path(folder).iterdir(), key=os.path.getmtime)[-1]
            print("Loaded model ",model)
            return model
        elif path.endswith("pt"):
            return path
        elif os.path.isdir(path):
            model = sorted(Path(path).iterdir(), key=os.path.getmtime)[-1]
            return self.get_model_path(model)
        return path


    def get_model(self):
        return self.get_type_model()

    def get_type_model(self):
        m_type = self.model_config['name']
        multichannel = self.data['datagen_config']['multiple_channel']['multiple_channel_len']
        if m_type == "classificator":
            return Classificator(self.model_config, pretrained=self.pretrained_model, 
            window_size=self.data['tokenizer']['window_size'], multichannel_len=multichannel, channels_in=1, class_dim=self.class_dim)
        raise Warning("No type model found")


    def get_optimizer(self, model):
        self.LEARNING_RATE = self.data["optimizer"]['learning_rate']
        w_decay = self.data["optimizer"]['w_decay']
        momentum = self.data["optimizer"]['momentum']
        loss_name = self.data["optimizer"]['name'].lower()
        if loss_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.LEARNING_RATE, weight_decay=w_decay)
        elif loss_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.LEARNING_RATE, weight_decay=w_decay, momentum=momentum)
        elif loss_name == "adagrad":
            return torch.optim.Adagrad(model.parameters(), lr=self.LEARNING_RATE, weight_decay=w_decay)
        elif loss_name == "adadelta":
            return torch.optim.Adadelta(model.parameters(), lr=self.LEARNING_RATE, weight_decay=w_decay)
        elif loss_name == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=self.LEARNING_RATE, weight_decay=w_decay)
        elif loss_name == "rprop":
            return torch.optim.Rprop(model.parameters(), lr=self.LEARNING_RATE, weight_decay=w_decay)
        elif loss_name == "lbfgs":
            return torch.optim.LBFGS(model.parameters(), lr=self.LEARNING_RATE, weight_decay=w_decay)
        raise Warning("No optimizer " + loss_name)


    def calculate_metric(self, output, targets):
        o_labels = torch.argmax(output, 1)
        print("Predicted Labels")
        logging.info("Predicted Labels")
        print(o_labels)
        logging.info(str(o_labels))
        n_correct = (o_labels==targets).sum().item()
        return n_correct
    
    def set_folders(self, exp_name):
        if not os.path.exists(self.data["model"]["folder_save"]):
            os.mkdir(self.data["model"]["folder_save"])
        base = self.data["model"]["folder_save"]
        self.base_path = os.path.join(base,exp_name)
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path )      
        self.plot_path = os.path.join(base,exp_name,"plots")
        if not os.path.exists(self.plot_path):
            os.mkdir(self.plot_path)
        

    def evaluate_model(self, epoch):
        n = 1
        n_correct = 0
        examples = 0
        for _,data in enumerate(self.validation_loader, 0):
            input = data["input"].to(self.device)
            target = data["output"].to(self.device)
            output = self.model(input)
            loss = self.loss_function(output, target)
            n_correct += self.calculate_metric(output, target)
            examples += target.size(0)
            if n % 50000 == 0:
                break
            n += 1
        accuracy = n_correct/examples
        self.dA_eval = accuracy - self.dA_eval
        self.evaluation_accuracy = accuracy
        print ("Accuracy: ", n_correct/examples)
        logging.info("Acc {}".format((n_correct/examples)*100))
        self.writer.add_scalar("Accuracy/Validation", n_correct/examples, epoch)
        torch.cuda.empty_cache()
        return

    def model_train(self, epoch):
        #print("Memory Model Used ", torch.cuda.memory_allocated(0)/1e+6)
        n = 1
        n_correct = 0
        examples = 0
        for _,data in enumerate(self.training_loader, 0):
            input1 = data["input"].to(self.device)
            target = torch.squeeze(data["output"],0).to(self.device)
            output = self.model(input1)
            #print("Targets ",target)
            loss_contrastive = self.loss_function(output, target)
            self.optimizer.zero_grad()
            loss_contrastive.backward()
            self.optimizer.step()
            logging.info("Epoch {} Current loss {}\n".format(epoch,loss_contrastive.item()))
            print("Epoch {} Current loss {}\n".format(epoch,loss_contrastive.item()))
            self.writer.add_scalar("Loss/train", loss_contrastive.item(), epoch)
            n_correct += self.calculate_metric(output, target)
            examples += target.size(0)
            logging.info("Target ")
            print("Target ")
            logging.info(str(target))
            print(target)
            logging.info("Acc {}".format((n_correct/examples)*100))
            print("Acc ", (n_correct/examples)*100)
            self.writer.add_scalar("Accuracy/train", (n_correct/examples)*100, epoch)
            n += 1
        accuracy = n_correct/examples
        self.dA_train = accuracy - self.dA_train
        self.train_accuracy = accuracy
        return


    def get_embeeddings_model(self):
        #print("Memory Model Used ", torch.cuda.memory_allocated(0)/1e+6)
        n = 1
        n_correct = 0
        examples = 0
        train_x = []
        train_y = []
        for _,data in enumerate(self.training_loader, 0):
            input1 = data["input"].to(self.device)
            target = torch.squeeze(data["output"],0).to(self.device)
            output = self.model.forward_pretrained(input1)
            train_x += output.cpu().detach().numpy().tolist()
            train_y += target.cpu().detach().numpy().tolist()
        return train_x, train_y

    def get_best_knn_model(self, df, y_v):
        values = {}
        train_x, train_y = self.get_embeeddings_model()
        best_accuracy = 0.0 
        best_model = {}
        for n_neighbors in range(1, 31):
            print("Evaluating KNN n_neighbors: ", n_neighbors)
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(train_x, train_y)
            y_predict = self.test_zero_shot(knn, df)
            classification = classification_report(y_v, y_predict, output_dict=True)
            accuracy = classification['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model.update({
                    'model': knn,
                    'n_neighbors': n_neighbors,
                    'y_predict': y_predict,
                    'accuracy': accuracy
                })
        return best_model

    def train_zero_shot(self, train_x, train_y, n_neighbors=5):
        classificator = KNeighborsClassifier(n_neighbors=n_neighbors)
        classificator.fit(train_x, train_y)
        return classificator
        
    def prepare_model(self):
        self.model = self.get_model()
        self.model_name = self.get_model_name()
        self.folder = self.base_path
        self.ext = self.data["model"]["extention"]
        self.full_path = os.path.join(self.folder, self.model_name + '.' + self.ext)        
        self.model.to(self.device)
        self.get_loss_function()
        self.optimizer = self.get_optimizer(self.model)

    def get_loss_function(self):
        loss_func = self.data["loss"]["loss_function"]
        if loss_func == "cross_entropy":
            self.loss_function = CrossEntropyLoss()
        elif loss_func == "mce":
            self.loss_function = MSELoss()
        else:
            raise Warning("No loss function")

    def train(self):
        torch.set_grad_enabled(True)
        self.EPOCHS=self.data["train"]["epochs"]
        self.model.train()
        log_dir = self.plot_path if self.save else 'runs'
        self.writer = SummaryWriter(log_dir=log_dir)
        torch.cuda.empty_cache()
        for epoch in range(self.EPOCHS):
            self.model_train(epoch)
            if self.data["train"]["save_each"] == "epoch":
                if self.save:
                    torch.save(self.model, "{}/{}-epoch-{}.{}".format(self.folder, self.model_name, epoch, self.ext))
                    print("Model Saved Successfully")
            if self.use_validation and self.data["validation"]["use_each"] == "epoch":
                torch.cuda.empty_cache()
                self.evaluate_model(epoch)
        if self.save:
            torch.save(self.model, self.full_path)

    def get_real_label(self, targets):
        if len(targets.shape) == 1:
            return [target.item() for target in targets]
        return [target.item() for target in targets.unsqueeze(0)]

    def get_num_samples(self, targets):
        if len(targets.shape) == 1:
            return targets.size(0)
        return targets.unsqueeze(0).size(0)

    def get_norm_targets(self, targets):
        if len(targets.shape) == 1:
            return targets
        return targets.unsqueeze(0)

    def add_embedding(self, embeddings, class_label):
        try:
            self.writer.add_embedding(embeddings, metadata=class_label)
        except:
            pass

    def fix_channels(self, channels):
        
        if len(channels.shape) == 1:
            return channels
        return channels.unsqueeze(0)

    def create_test_data_output(self, df, data, output1):
        n_batch = output1.size(0)
        g_value = lambda x: x.item() if isinstance(x, torch.Tensor) else x
        for b in range(n_batch):
            if not self.multichannel_create:
                df.append({'Vector': output1.to("cpu").detach().numpy()[b], "Categ": g_value(data["output"][b]), "subject": g_value(data["subject"][b]), "chn": g_value(data["channels"][b]) })
                #df.append({'Vector': output2.to("cpu").detach().numpy()[b], "Categ": g_value(data["output"][b]), "subject": g_value(data["subject2"][b]), "chn": g_value(data["chn2"][b]), "estimulo": g_value(data["estimulo"][b])})
                #self.add_embedding(output1, output2, g_value(data["output"][b]), b)
            else:
                df.append({'Vector': output1.to("cpu").detach().numpy()[b], "Categ": data["output"][b].item(), "subject": data["subject"].squeeze()[b].item(), "chn": len(data["channels"]), "channels": "all"})
                #df.append({'Vector': output2.to("cpu").detach().numpy()[b], "Categ": data["output"][b].item(), "subject": data["subjects"][0][b].item(), "chn": len(data["channels"]), "channels": [chn[b].item() for chn in data["channels"]], "estimulo": data["stimulus"][0][b].item()})
                #self.add_embedding(output1, output2, data["output"][b].item(), b)

    def test(self):
        y_real = []
        y_predict = []
        n_correct = 0
        examples = 0
        df = list()
        datalen = len(self.testing_set)
        for _,data in enumerate(self.testing_loader, 0):
            input = data["input"].to(self.device)
            targets = torch.squeeze(data["output"],0).to(self.device)
            output = self.model(input)
            loss_contrastive = self.loss_function(output, targets)
            y_real += self.get_real_label(targets)
            labels_predict = output.squeeze(0).to("cpu").detach().numpy()
            y_predict += np.argmax(labels_predict, 1).tolist()
            n_correct += self.calculate_metric(output, targets)
            examples += self.get_num_samples(targets)
            logging.info("Acc {}".format((n_correct/examples)*100))
            print("Acc {}".format((n_correct/examples)*100))
            if psutil.virtual_memory()[2] > MEMORY_LIMIT:
                print("Memory is full")
                break
            self.create_test_data_output(df, data, self.model.forward_pretrained(input))
            logging.info("Completado: " + str((examples/datalen)*100))
            print("Completado: ", (examples/datalen)*100)
        self.test_accuracy = (n_correct/examples)
        torch.cuda.empty_cache()
        return y_real, y_predict, DataFrame.from_dict(df)

    def test_zero_shot(self, classificator, df):
        y_predict = classificator.predict(df.Vector.values.tolist())
        return y_predict


    def test_optuna(self):
        n_correct = 0
        examples = 0
        datalen = len(self.testing_set)
        for _,data in enumerate(self.testing_loader, 0):
            input = data["input"].to(self.device)
            targets = torch.squeeze(data["output"],0).to(self.device)
            output = self.model(input)
            loss_contrastive = self.loss_function(output, targets)
            n_correct += self.calcuate_metric(output, targets)
            examples += self.get_num_samples(targets)
            logging.info("Acc {}".format((n_correct/examples)*100))
            print("Acc {}".format((n_correct/examples)*100))
            #self.create_test_data_output(df, data, output1, output2)
            logging.info("Completado: " + str((examples/datalen)*100))
            print("Completado: ", (examples/datalen)*100)
        self.test_accuracy = (n_correct/examples)
        torch.cuda.empty_cache()
        return

    def save_crosstab(self, Y_V, Y_P):
        confusion_matrix = crosstab(Y_V, Y_P, rownames=['Real'], colnames=['Predicción'])
        confusion_matrix.to_csv(os.path.join(self.plot_path, "crosstab.csv"), mode='w')

    def save_classification_report(self, Y_V, Y_P):
        class_report = classification_report(Y_V, Y_P)
        report = open(os.path.join(self.plot_path,"report.txt"),"w")
        report.write(class_report)
        report.close()

    def gen_model_reports(self, y_real, y_predict, df):
        Y_P = np.array(y_predict)
        Y_V = np.array(y_real)
        self.save_crosstab(Y_V, Y_P)
        folder = self.plot_path
        self.save_classification_report(Y_V, Y_P)
        frac = self.data["test"]["dataset_frac"]
        #self.add_embeddings(df)
        # TSNE
        df_plot = df.sample(frac=frac)
        #df_plot = df
        self.plot_tsne(df_plot)
        #UMAP
        self.plot_umap_proc(df_plot)
        #self.save_test_data(df)
        self.get_silhouette_result(df)
        return

    def add_embeddings(self, df_plot):
        self.add_embedding(np.array(df_plot.Vector.tolist()), df_plot.Categ)
        self.add_embedding(np.array(df_plot.Vector.tolist()), df_plot.subject)
        self.add_embedding(np.array(df_plot.Vector.tolist()), df_plot.chn)
        self.add_embedding(np.array(df_plot.Vector.tolist()), df_plot.estimulo)

    def get_data_model_report(self):
        folder = self.plot_path
        report = open(os.path.join(folder,"data-model.txt"),"w")
        data = """
        Entrenamiento:
        {}
        Test:
        {}
        Validacion:
        {}

        MODELO:
        {}
        """.format( str(self.train_data_generator.dataset_metadata), 
                    str(self.test_data_generator.dataset_metadata),
                    str(self.validation_data_generator.dataset_metadata if self.use_validation else "No hay validacion"),
                    str(self.model.shapes))
        report.write(data)
        report.close()   
        return

    def get_silhouette_result(self, df):
        folder = self.plot_path
        report = open(os.path.join(folder,"silhouette.txt"),"w")
        try:
            score_category = silhouette_score(np.array(df.Vector.tolist()), np.array(df.Categ.tolist()))
        except ValueError:
            score_category = -999999
        try:
            score_subject = silhouette_score(np.array(df.Vector.tolist()), np.array(df.subject.tolist()))
        except ValueError:
            score_subject = -999999
        if self.model_config['name'] in ["siamese_conv", "siamese_linear"]:
            try:
                score_channel = silhouette_score(np.array(df.Vector.tolist()), np.array(df.chn.tolist()))
            except ValueError:
                score_channel = -999999
        else:
            score_channel = -999999999
        report_txt = """
        Silhouette 
        Score Category: {}
        Score Subject:  {}
        Score Channel:  {}
        """.format(score_category,score_subject,score_channel)
        report.write(report_txt)
        report.close()   
        return

    def save_test_data(self,df):
        path = os.path.join(self.base_path, 'data_test.csv')
        pickle_path =  os.path.join(self.base_path, 'data_test.bin')
        pickle_file = open(pickle_path, "wb")
        pickle.dump(df, pickle_file)
        return df.to_csv(path)

    def plot_umap_proc(self, df):
        folder = self.plot_path
        umap_2d = UMAP(n_components=2, spread=1, min_dist=0.5, a=0.7, b=1.2)
        umap_3d = UMAP(n_components=3, spread=1, min_dist=0.5, a=0.7, b=1.2)
        proj_2d = umap_2d.fit_transform(np.array(df.Vector.tolist()))
        proj_3d = umap_3d.fit_transform(np.array(df.Vector.tolist()))
        self.plot_umap(folder,proj_2d,proj_3d,df.Categ,"Categ","category-umap")
        self.plot_umap(folder,proj_2d,proj_3d,df.subject,"subject","subject-umap")
        self.plot_umap(folder,proj_2d,proj_3d,df.chn,"chn","channel-umap")
        #self.plot_umap(folder,proj_2d,proj_3d,df.estimulo,"estimulo","estimulo-umap")

    def plot_umap(self, folder, proj_2d, proj_3d, data, label, basename):
        fig_2d = px.scatter(proj_2d, x=0, y=1, color=data, labels={'color': label}).write_image(self.get_plot_name(folder,basename,"2d"))
        fig_3d = px.scatter_3d(proj_3d, x=0, y=1, z=2,color=data, labels={'color': label}).write_image(self.get_plot_name(folder,basename,"3d"))
        return 

    def plot_tsne(self, df):
        folder = self.plot_path
        perplexity_heu = int(np.sqrt(df.shape[0]))
        m = TSNE(n_components=3, perplexity=perplexity_heu, n_iter=1000, n_jobs=-1)
        tsne_features = m.fit_transform(np.array(df.Vector.tolist()))
        df["x"] = tsne_features[:,0]
        df["y"] = tsne_features[:,1]
        df["z"] = tsne_features[:,2]
        #m2d = TSNE(n_components=2, perplexity=perplexity_heu, n_iter=1000)
        #tsne_features = m2d.fit_transform(np.array(df.Vector.tolist()))
        #df["x2d"] = tsne_features[:,0]
        #df["y2d"] = tsne_features[:,1]
        self.plot(folder, df, "Categ", df.Categ, "category-tsne")
        self.plot(folder, df, "subject", df.subject, "subject-tsne")
        self.plot(folder, df, "chn", df.chn, "channel-tsne")
        #self.plot(folder, df, "estimulo", df.estimulo, "estimulo-tsne")

    def get_plot_name(self, folder, base, extra):
        return os.path.join(folder,"{}{}.png".format(base,extra))

    def plot(self, folder, df, hue_data, data, basename):
        #sns.scatterplot(x="x",y="y", hue=hue_data, data=df).figure.savefig(self.get_plot_name(folder,basename,""))
        px.scatter_3d(df, x='x', y='y', z='z', color=data).write_image(self.get_plot_name(folder,basename,"3d"))
        px.scatter(df, x='x', y='y', color=data).write_image(self.get_plot_name(folder,basename,"2d"))

    def plot_channels(self, folder, df):
        #sns.scatterplot(x="x",y="y", hue="chn", data=df).savefig(os.path.join(folder,"channel.png"))
        fig3 = px.scatter_3d(df, x='x', y='y', z='z',color=df.chn)
        fig3_2d = px.scatter(df, x='x', y='y',color=df.chn)
        fig3_2d.write_image(os.path.join(folder,"channel2d.png"))
        fig3.write_image(os.path.join(folder,"channel3d.png"))

    def plot_subject(self, folder, df):
        #sns.scatterplot(x="x",y="y", hue="subject", data=df).savefig(os.path.join(folder,"subject.png"))
        fig2 = px.scatter_3d(df, x='x', y='y', z='z',color=df.subject)
        fig2_2d = px.scatter(df, x='x', y='y',color=df.subject)
        fig2_2d.write_image(os.path.join(folder,"subject2d.png"))
        fig2.write_image(os.path.join(folder,"subject3d.png"))

    def plot_category(self, folder, df):
        #sns.scatterplot(x="x",y="y", hue="Categ", data=df).savefig(os.path.join(folder,"category.png"))
        fig = px.scatter_3d(df, x='x', y='y', z='z',color=df.Categ)
        fig_2d = px.scatter(df, x='x', y='y', color=df.Categ)
        fig_2d.write_image(os.path.join(folder,"category2d.png"))
        fig.write_image(os.path.join(folder,"category3d.png"))

    def save_yaml_conf(self):
        path = os.path.join(self.base_path, 'data.yaml') 
        with open(path, 'w') as outfile:
            yaml.dump(self.data, outfile, default_flow_style=False)
        path_model = os.path.join(self.base_path, 'model_config.yaml') 
        # Se lee otra vez el archivo de configuracion por un bug
        self.model_config = self._get_dict_model_file(self.data["model"]["model_config_path"])
        #with open(self.data["model"]["model_config_path"]) as f:
        #    self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        #######################################
        with open(path_model, 'w') as outfile:
            yaml.dump(self.model_config , outfile, default_flow_style=False)        
        

    def run_optuna(self):
        self.save = False
        self.set_model_config()
        self.model = self.get_model()     
        self.model.to(self.device)
        self.get_loss_function()
        self.optimizer = self.get_optimizer(self.model)
        if not self.data['only_test']:
            self.train()
            self.model.eval()
        y_real, y_predict, df =self.test()
        values = self.get_best_knn_model(df, y_real)
        self.save_knn_data(values)
        self.gen_model_reports(y_real, y_predict, df)
        #self.test_optuna()

    def save_knn_data(self, data):
        new_data = {
            'n_neighbors': data['n_neighbors'],
            'accuracy': data['accuracy']
        }
        self.knn_data = new_data
        path_model = os.path.join(self.plot_path, 'knn.json') 
        knn = open(path_model, 'w')
        return json.dump(new_data, knn)

    def save_model(self, folder, name="checkpoint"):
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.model, "{}/{}.{}".format(folder, name, self.data["model"]["extention"]))

    def handle_optim_knn(self, y_real, y_predict, df):
        values = self.get_best_knn_model(df, y_real)
        print("BEST MODEL n_neighbors: ", values['n_neighbors'])
        print("BEST MODEL accuracy: ", values['accuracy'])
        self.save_knn_data(values)
        return self.gen_model_reports(y_real, values['y_predict'], df)

    def run(self):
        self.save = True
        self.set_model_config()
        branch = "emotion_neural_interface_{}_dev-{}_{}".format(self.get_model_name(),self.data["github"]["dev"], datetime.now().timestamp())
        self.data["branch"] = branch
        self.set_folders(branch)
        self.save_yaml_conf()
        self.prepare_model()
        self.get_data_model_report()
        if not self.data['only_test']:
            self.train()
            self.model.eval()
        y_real, y_predict, df =self.test()
        if self.data['use_zero_shot']:
            train_x, train_y = self.get_embeeddings_model()
            n_neighbors = self.data['knn']['n_neighbors']
            if n_neighbors == "optim":
                return self.handle_optim_knn(y_real, y_predict, df)
            classificator = self.train_zero_shot(train_x, train_y, n_neighbors=n_neighbors)
            y_predict = self.test_zero_shot(classificator, df)
            self.gen_model_reports(y_real, y_predict, df)
        else:
            self.gen_model_reports(y_real, y_predict, df)
        
    def run_test(self):
        branch = "emotion_neural_interface_{}_dev-{}_{}".format(self.get_model_name(),self.data["github"]["dev"], datetime.now().timestamp())
        self.data["branch"] = branch
        self.set_folders(branch)
        self.save_yaml_conf()
        self.prepare_model()
        self.get_data_model_report()
        if not self.data['use_pretrained']:
            self.train()
            self.model.eval()
        y_real, y_predict, df =self.test()
        if self.data['use_zero_shot']:
            train_x, train_y = self.get_embeeddings_model()
            classificator = self.train_zero_shot(train_x, train_y)
            y_predict = self.test_zero_shot(classificator, df)
            self.gen_model_reports(y_real, y_predict, df)
        else:
            self.gen_model_reports(y_real, y_predict, df)

if __name__ == '__main__':
    exp = FinetuneModel("config/config_finetuning.yaml")
    exp.run()
