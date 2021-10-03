from EmotionNeuralInterface.tools.paths_utils import get_paths_experiment, get_path
from EmotionNeuralInterface.subject_data.utils import create_subject_data
from EmotionNeuralInterface.data.tokenizer import Tokenizer
#from EmotionNeuralInterface.data.datagen import DataGen
# Datasets Generators
from EmotionNeuralInterface.data.pretext_task.same_channel_single_channel import SameChannel
from EmotionNeuralInterface.data.pretext_task.same_subject_single_channel import SameSubject
from EmotionNeuralInterface.data.pretext_task.consecutive_single_channel import Consecutive
# Pretext Tasks
from EmotionNeuralInterface.data.pretext_task.relative_positioning import RelativePositioning
from EmotionNeuralInterface.data.pretext_task.temporal_shifting import TemporalShifting
# Tools
from EmotionNeuralInterface.data.dataset import NetworkDataSet
from EmotionNeuralInterface.tools.utils import split_data_by_len
# Loss
from EmotionNeuralInterface.model.loss import ContrastiveLoss
from EmotionNeuralInterface.model.loss import NTXentLoss
from EmotionNeuralInterface.model.loss import CosineEmbeddingLoss
from EmotionNeuralInterface.model.loss import ContrastiveLossSameZero
# Models
from EmotionNeuralInterface.model.model import SiameseLinearNetwork
from EmotionNeuralInterface.model.model import SiameseNetwork
from EmotionNeuralInterface.model.stage_net import StageNet
from EmotionNeuralInterface.model.nedbert import NedBERT

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

class Workbench(object):
    def __init__(self, config_file):
        with open(config_file) as f:
            self.data = yaml.load(f, Loader=yaml.FullLoader)
            print(self.data)
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        torch.set_grad_enabled(False)
        self.model_config = {}
        self.load_dataset()
        self.load_tokenizer()
        self.dataset_subjects()
        self.datagen_config()
        self.load_dataset_batch()
    
    def load_dataset(self):
        path = get_path(self.data["dataset"]["dataset_path"])
        experiments_paths = get_paths_experiment(path)
        self.experiments, self.subjects = create_subject_data(experiments_paths)
        
    def load_tokenizer(self):
        data = self.data["tokenizer"]
        self.tokenizer = Tokenizer(self.subjects, window_size=data["window_size"], stride=data["stride"])

    def dataset_subjects(self):
        data = self.data["dataset_subjects"]
        self.train_subjets, other_subjets = split_data_by_len(self.subjects.copy_list(), data["train_subjects"])
        self.validation_subjets, self.test_subjets = split_data_by_len(other_subjets,data["test_subjects"])
    
    def datagen_config(self):
        data = self.data["datagen_config"]
        self.target_cod = data["target_codification"]
        self.combinate_subjects = data["combinate_subjects"]
        self.channel_iters = data["channel_iters"]
        # Dataset Len
        self.dataset_train_len = data['dataset_train_len']
        self.dataset_test_len = data['dataset_test_len']
        self.dataset_validation_len = data['dataset_validation_len']
        if data["use_overfitting"]:
            self.datagen_overfitting()
            return
        self.set_train_datagen()
        self.set_validation_datagen()
        self.set_test_datagen()

    def set_train_datagen(self):
        self.train_data_generator = self.get_dataset_generator(self.train_subjets, dataset_max_len=self.dataset_train_len)
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
        if self.data["datagen_config"]["dataset"] == "same_channel_single_channel":
            return SameChannel(subjects, self.tokenizer, combinate_subjects=self.combinate_subjects, channels_iter=self.channel_iters, max_data=dataset_max_len, targets_cod=self.target_cod)
        elif self.data["datagen_config"]["dataset"] == "same_subject_single_channel":
            return SameSubject(subjects, self.tokenizer, combinate_subjects=self.combinate_subjects, channels_iter=self.channel_iters, max_data=dataset_max_len, targets_cod=self.target_cod)
        elif self.data["datagen_config"]["dataset"] == "consecutive_single_channel":
            return Consecutive(subjects, self.tokenizer, combinate_subjects=self.combinate_subjects, channels_iter=self.channel_iters, max_data=dataset_max_len, targets_cod=self.target_cod)
        elif self.data["datagen_config"]["dataset"] == "relative_positioning_multiple_channel":
            return RelativePositioning(subjects, self.tokenizer, multiple_channel_len=multiple_channel_dict["multiple_channel_len"],
                                        t_pos_max=multiple_channel_dict["t_pos_max"],dataset_len=dataset_max_len,
                                        max_num_iter=multiple_channel_dict["max_num_iter"], targets_cod=self.target_cod)
        elif self.data["datagen_config"]["dataset"] == "temporal_shifting_multiple_channel":
            return TemporalShifting(subjects, self.tokenizer, multiple_channel_len=multiple_channel_dict["multiple_channel_len"],
                                        t_pos_max=multiple_channel_dict["t_pos_max"],dataset_len=dataset_max_len,
                                        max_num_iter=multiple_channel_dict["max_num_iter"],targets_cod=self.target_cod)
        
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

        train_params = {'batch_size': self.TRAIN_BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 0
                        }

        validation_params = {'batch_size': self.VALID_BATCH_SIZE,
                            'shuffle': True,
                            'num_workers': 0
                        }

        test_params = {'batch_size': self.TEST_BATCH_SIZE,
                            'shuffle': True,
                            'num_workers': 0
                        }
        self.training_loader = DataLoader(self.training_set, **train_params)
        self.validation_loader = DataLoader(self.validation_set, **validation_params)
        self.testing_loader = DataLoader(self.testing_set, **test_params)
     
    def get_model_features(self):
        return self.model_config['name']

    def get_model_name(self):
        return "model-{}-{}-margin{}-P{}-loss{}-epoch{}".format(self.get_model_features(), 
            self.data["datagen_config"]["dataset"], self.data["loss"]["margin"], self.target_cod["positive"],
            self.data["loss"]["loss_function"], self.data["train"]["epochs"])

    def set_model_config(self):
        with open(self.data["model"]["model_config_path"]) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
            print(self.model_config)

    def get_model(self):
        if self.data["train"]["load_model"]:
            return torch.load(self.data["train"]["load_model_name"])
        return self.get_type_model()

    def get_type_model(self):
        m_type = self.model_config['name']
        if m_type == "siamese_stagenet":
            channels = self.data["datagen_config"]["multiple_channel"]["multiple_channel_len"]
            return StageNet(self.model_config, width=self.data["tokenizer"]["window_size"],height=channels)
        elif m_type == "siamese_conv":
            return SiameseNetwork(self.model_config, window_size=self.data["tokenizer"]["window_size"])
        elif m_type == "siamese_linear":
            return SiameseLinearNetwork(self.model_config, window_size=self.data["tokenizer"]["window_size"])
        elif m_type == "nedbert":
            return NedBERT(self.model_config, sequence_lenght=self.data["tokenizer"]["window_size"])
        raise Warning("No type model found")


    def get_optimizer(self, model):
        self.LEARNING_RATE = self.data["optimizer"]['learning_rate']
        w_decay = self.data["optimizer"]['w_decay']
        momentum = self.data["optimizer"]['momentum']
        if self.data["optimizer"]['name'] == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.LEARNING_RATE, weight_decay=w_decay)
        elif self.data["optimizer"]['name'] == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.LEARNING_RATE, weight_decay=w_decay, momentum=momentum)
        raise Warning("No optimizer")


    def calculate_distance(self, output1, output2):
        fn = self.get_distance_function()
        return fn(output1, output2)
        #return torch.sqrt((output2 - output1).pow(2).sum(1))


    def get_distance_function(self):
        if self.data["loss"]["loss_function"] in ["NTXentLoss", "CosineEmbeddingLoss"]: 
            return F.cosine_similarity
        return F.pairwise_distance

    def calculate_label(self, distances):
        return torch.cuda.IntTensor([self.target_cod["positive"] if tensor.item() < self.margin else self.target_cod["negative"] for tensor in distances])

    def calcuate_metric(self, output1, output2, targets):
        batches = output1.size(0)
        distances = self.calculate_distance(output1, output2)
        o_labels = self.calculate_label(distances)
        print("Predicted Labels")
        print(o_labels)
        n_correct = (o_labels==targets).sum().item()
        return n_correct
    
    def set_folders(self, exp_name):
        if not os.path.exists(self.data["model"]["folder_save"]):
            os.mkdir(self.data["model"]["folder_save"])
        base = self.data["model"]["folder_save"]
        self.base_path = os.path.join(base,exp_name)
        if not os.path.exists(self.base_path ):
            os.mkdir(self.base_path )      
        self.plot_path = os.path.join(base,exp_name,"plots")
        if not os.path.exists(self.plot_path):
            os.mkdir(self.plot_path)
        

    def evaluate_model(self, epoch):
        n = 1
        n_correct = 0
        examples = 0
        for _,data in enumerate(self.validation_loader, 0):
            input1 = data["input1"].to(self.device)
            input2 = data["input2"].to(self.device)
            target = data["output"].to(self.device)
            output1, output2 = self.model(input1, input2)
            loss_contrastive = self.loss_function(output1, output2, target.unsqueeze(1))
            distance = self.calculate_distance(output1, output2)
            n_correct += self.calcuate_metric(output1, output2, target)
            examples += target.size(0)
            if n % 10000 == 0:
                break
            n += 1
        print ("Accuracy: ", n_correct/examples)
        self.writer.add_scalar("Accuracy/Validation", n_correct/examples, epoch)
        return

    def model_train(self, epoch):
        n = 1
        n_correct = 0
        examples = 0
        distance = self.get_distance_function()
        for _,data in enumerate(self.training_loader, 0):
            input1 = data["input1"].to(self.device)
            input2 = data["input2"].to(self.device)
            target = torch.squeeze(data["output"],0).to(self.device)
            output1, output2 = self.model(input1, input2)
            #print("Targets ",target)
            loss_contrastive = self.loss_function(output1, output2, target)
            self.optimizer.zero_grad()
            loss_contrastive.backward()
            self.optimizer.step()
            print("Epoch {} Current loss {}\n".format(epoch,loss_contrastive.item()))
            self.writer.add_scalar("Loss/train", loss_contrastive.item(), epoch)
            _distance = distance(output1, output2)
            n_correct += self.calcuate_metric(output1, output2, target)
            examples += target.size(0)
            print("Distance ")
            print(_distance)
            print("Target ")
            print(target)
            print("Acc ", (n_correct/examples)*100)
            self.writer.add_scalar("Accuracy/train", (n_correct/examples)*100, epoch)
            n += 1
        return

    def prepare_model(self):
        self.model = self.get_model()
        self.model_name = self.get_model_name()
        self.folder = self.base_path
        self.ext = self.data["model"]["extention"]
        self.full_path = os.path.join(self.folder, self.model_name + self.ext)        
        self.model.to(self.device)
        self.get_loss_function()
        self.optimizer = self.get_optimizer(self.model)

    def get_loss_function(self):
        loss_func = self.data["loss"]["loss_function"]
        if loss_func == "contrastive_loss":
            self.margin = self.data["loss"]["margin"]
            self.loss_function = ContrastiveLoss(self.margin)
        elif loss_func == "margin_raking":
            self.margin = self.data["loss"]["margin"]
            self.loss_function = nn.MarginRankingLoss(margin=self.margin)
        elif loss_func == "NTXentLoss":
            batch_size = self.data['dataset_batch']['train_batch_size']
            self.loss_function = NTXentLoss(self.device, batch_size, self.data['loss']['temperature'])
        elif loss_func == "CosineEmbeddingLoss":
            self.margin = self.data["loss"]["margin"]
            self.loss_function = CosineEmbeddingLoss(self.margin)
        elif loss_func == "contrastive_loss_custom":
            self.margin = self.data["loss"]["margin"]
            self.loss_function = ContrastiveLossSameZero(self.margin)
        else:
            raise Warning("No loss function")

    def train(self):
        #import pdb;pdb.set_trace()
        self.prepare_model()
        torch.set_grad_enabled(True)
        self.EPOCHS=self.data["train"]["epochs"]
        self.model.train()
        self.writer = SummaryWriter(log_dir=self.plot_path)
        for epoch in range(self.EPOCHS):
            self.model_train(epoch)
            if self.data["train"]["save_each"] == "epoch":
                torch.save(self.model, "{}/{}-epoch-{}.{}".format(self.folder, self.model_name, epoch, self.ext))
                print("Model Saved Successfully")
            if self.data["validation"]["use_each"] == "epoch":
                self.evaluate_model(epoch)
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


    def add_embedding(self, output1, output2, meta, b):
        try:
            self.writer.add_embedding(output1.to("cpu").detach().numpy()[b], metadata=[meta])
            self.writer.add_embedding(output2.to("cpu").detach().numpy()[b], metadata=[meta])
        except:
            pass

    def create_test_data_output(self, df, data, output1, output2):
        #import pdb;pdb.set_trace()
        n_batch = output1.size(0)
        g_value = lambda x: x.item() if isinstance(x, torch.Tensor) else x
        for b in range(n_batch):
            if self.model_config['name'] in ["siamese_conv", "siamese_linear"]:
                df.append({'Vector': output1.to("cpu").detach().numpy()[b], "Categ": g_value(data["output"][b]), "subject": g_value(data["subject1"][b]), "chn": g_value(data["chn1"][b]), "estimulo": g_value(data["estimulo"][b])})
                df.append({'Vector': output2.to("cpu").detach().numpy()[b], "Categ": g_value(data["output"][b]), "subject": g_value(data["subject2"][b]), "chn": g_value(data["chn2"][b]), "estimulo": g_value(data["estimulo"][b])})
                self.add_embedding(output1, output2, g_value(data["output"][b]), b)
            else:
                df.append({'Vector': output1.to("cpu").detach().numpy()[b], "Categ": data["output"][b].item(), "subject": data["subjects"][0][b].item(), "chn": len(data["channels"]), "channels": [chn[b].item() for chn in data["channels"]], "estimulo": data["stimulus"][0][b].item()})
                df.append({'Vector': output2.to("cpu").detach().numpy()[b], "Categ": data["output"][b].item(), "subject": data["subjects"][0][b].item(), "chn": len(data["channels"]), "channels": [chn[b].item() for chn in data["channels"]], "estimulo": data["stimulus"][0][b].item()})
                self.add_embedding(output1, output2, data["output"][b].item(), b)

    def test(self):
        y_real = []
        y_predict = []
        y_distance = []
        n_correct = 0
        examples = 0
        df = list()
        datalen = len(self.training_set)
        i = 1
        for _,data in enumerate(self.testing_loader, 0):
            input1 = data["input1"].to(self.device)
            input2 = data["input2"].to(self.device)
            targets = torch.squeeze(data["output"],0).to(self.device)
            output1, output2 = self.model(input1, input2)
            loss_contrastive = self.loss_function(output1, output2, targets)
            #eucledian_distance = F.pairwise_distance(output1, output2)
            y_real += self.get_real_label(targets)
            labels_predict = self.calculate_label(self.calculate_distance(output1, output2))
            y_predict += [label.item() for label in labels_predict]
            n_correct += self.calcuate_metric(output1, output2, targets)
            examples += self.get_num_samples(targets)
            print("Acc= ", n_correct/examples)
            self.create_test_data_output(df, data, output1, output2)
            print("Completado: ", i)
            i += 1
        return y_real, y_predict, y_distance, DataFrame.from_dict(df)

    def gen_model_reports(self, y_real, y_predict, y_distance, df):
        Y_P = np.array(y_predict)
        Y_V = np.array(y_real)
        confusion_matrix = crosstab(Y_P, Y_V, rownames=['Real'], colnames=['Predicción'])
        class_report = classification_report(Y_V, Y_P)
        folder = self.plot_path
        report = open(os.path.join(folder,"report.txt"),"w")
        report.write(class_report)
        report.close()
        #import pdb;pdb.set_trace()
        frac = self.data["test"]["dataset_frac"]
        # TSNE
        df_plot = df.sample(frac=frac)
        df_plot = df 
        self.plot_tsne(df_plot)
        #UMAP
        self.plot_umap_proc(df_plot)
        #self.save_test_data(df)
        self.get_silhouette_result(df)
        self.get_data_model_report()
        return
    
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
                    str(self.validation_data_generator.dataset_metadata),
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
        try:
            score_stimulus = silhouette_score(np.array(df.Vector.tolist()), np.array(df.estimulo.tolist()))
        except ValueError:
            score_stimulus = -999999999
        report_txt = """
        Silhouette 
        Score Category: {}
        Score Subject:  {}
        Score Channel:  {}
        Score Stimulus: {}
        """.format(score_category,score_subject,score_channel,score_stimulus)
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
        umap_2d = UMAP(n_components=2, init='random', random_state=0)
        umap_3d = UMAP(n_components=3, init='random', random_state=0)
        proj_2d = umap_2d.fit_transform(np.array(df.Vector.tolist()))
        proj_3d = umap_3d.fit_transform(np.array(df.Vector.tolist()))
        self.plot_umap(folder,proj_2d,proj_3d,df.Categ,"Categ","category-umap")
        self.plot_umap(folder,proj_2d,proj_3d,df.subject,"subject","subject-umap")
        self.plot_umap(folder,proj_2d,proj_3d,df.chn,"chn","channel-umap")
        self.plot_umap(folder,proj_2d,proj_3d,df.estimulo,"estimulo","estimulo-umap")

    def plot_umap(self, folder, proj_2d, proj_3d, data, label, basename):
        fig_2d = px.scatter(proj_2d, x=0, y=1, color=data, labels={'color': label}).write_image(self.get_plot_name(folder,basename,"2d"))
        fig_3d = px.scatter_3d(proj_3d, x=0, y=1, z=2,color=data, labels={'color': label}).write_image(self.get_plot_name(folder,basename,"3d"))
        return 

    def plot_tsne(self, df):
        folder = self.plot_path
        m = TSNE(n_components=3)
        tsne_features = m.fit_transform(np.array(df.Vector.tolist()))
        df["x"] = tsne_features[:,0]
        df["y"] = tsne_features[:,1]
        df["z"] = tsne_features[:,2]
        m2d = TSNE(n_components=2)
        tsne_features = m2d.fit_transform(np.array(df.Vector.tolist()))
        df["x2d"] = tsne_features[:,0]
        df["y2d"] = tsne_features[:,1]
        self.plot(folder, df, "Categ", df.Categ, "category-tsne")
        self.plot(folder, df, "subject", df.subject, "subject-tsne")
        self.plot(folder, df, "chn", df.chn, "channel-tsne")
        self.plot(folder, df, "estimulo", df.estimulo, "estimulo-tsne")

    def get_plot_name(self, folder, base, extra):
        return os.path.join(folder,"{}{}.png".format(base,extra))

    def plot(self, folder, df, hue_data, data, basename):
        #sns.scatterplot(x="x",y="y", hue=hue_data, data=df).figure.savefig(self.get_plot_name(folder,basename,""))
        px.scatter_3d(df, x='x', y='y', z='z', color=data).write_image(self.get_plot_name(folder,basename,"3d"))
        px.scatter(df, x='x2d', y='y2d', color=data).write_image(self.get_plot_name(folder,basename,"2d"))

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
        with open(self.data["model"]["model_config_path"]) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        #######################################
        with open(path_model, 'w') as outfile:
            yaml.dump(self.model_config , outfile, default_flow_style=False)        
        

    def run(self):
        self.set_model_config()
        branch = "emotion_neural_interface_{}_dev-{}_{}".format(self.get_model_name(),self.data["github"]["dev"], datetime.now().timestamp())
        self.data["branch"] = branch
        self.set_folders(branch)
        command = "git checkout -b {}".format(branch)
        os.system(command)
        self.train()
        self.model.eval()
        y_real, y_predict, y_distance, df =self.test()
        self.gen_model_reports(y_real, y_predict, y_distance, df)
        self.save_yaml_conf()
        os.system("git add .")
        os.system("git commit -m 'Se agrega el experimento'")
        os.system("git checkout main")

    def run_test(self):
        branch = "emotion_neural_interface_{}_dev-{}_{}".format(self.get_model_name(),self.data["github"]["dev"], datetime.now().timestamp())
        self.data["branch"] = branch
        self.set_folders(branch)
        command = "git checkout -b {}".format(branch)
        os.system(command)
        self.prepare_model()
        self.model.eval()
        y_real, y_predict, y_distance, df =self.test()
        self.gen_model_reports(y_real, y_predict, y_distance, df)
        self.save_yaml_conf()
        os.system("git add .")
        os.system('git commit -m "Se agrega el experimento"')
        os.system("git checkout main")    
        return
    
exp = Workbench("config/config.yaml")
exp.run()