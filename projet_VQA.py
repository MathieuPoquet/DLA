# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:30:28 2020

@author: Mathieu
"""

#%% Import des données

import torch

from torch.utils.data import dataloader
from torch.utils.data import Dataset

import torchvision.transforms as transforms

from PIL import Image

import pandas as pd

from typing import Any, Callable, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class VQADataset(Dataset):
  """
    This class loads a shrinked version of the VQA dataset (https://visualqa.org/)
    Our shrinked version focus on yes/no questions. 
    To load the dataset, we pass a descriptor csv file. 
    
    Each entry of the csv file has this form:

    question_id ; question_type ; image_name ; question ; answer ; image_id

  """
  def __init__(self, path : str, dataset_descriptor : str, image_folder : str, transform : Callable, vectorizer=None) -> None:
    """
      :param: path : a string that indicates the path to the image and question dataset.
      :param: dataset_descriptor : a string to the csv file name that stores the question ; answer and image name
      :param: image_folder : a string that indicates the name of the folder that contains the images
      :param: transform : a torchvision.transforms wrapper to transform the images into tensors 
    """
    super(VQADataset, self).__init__()
    self.descriptor = pd.read_csv(path + '/' + dataset_descriptor, delimiter=';')
    self.vectorizer = vectorizer
    self.modif_question(suppr_debut=True,suppr_interrogation=True)
    
    #self.tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    self.path = path 
    self.image_folder = image_folder
    self.transform = transform
    self.size = len(self.descriptor)
  
  def __len__(self) -> int:
    return self.size

  def __getitem__(self, idx : int) -> Tuple[Any, Any, Any]:
    """
      returns a tuple : (image, question, answer)
      image is a Tensor representation of the image
      question and answer are strings
    """
    image_name = self.path + '/' + self.image_folder + '/' + self.descriptor["image_name"][idx]

    image = Image.open(image_name).convert('RGB')

    image = self.transform(image)

    question = self.descriptor["question"][idx]
    
    #question_encode=self.tokenizer(question, truncation=True, padding=True)

    answer = self.descriptor["answer"][idx]
    
    answer_bool=1
    if answer=="no": answer_bool=0
            

    return (image, question, answer_bool)  #question_encode,

  def modif_question(self,suppr_debut = True,suppr_interrogation = True):
    corpus = self.descriptor["question"].to_numpy()
    
    corpus = modif_corpus(corpus,self.descriptor['question_type'],suppr_debut = suppr_debut,suppr_interrogation = True)
    
    X = self.vectorizer.fit_transform(corpus)
    X = X.toarray().tolist()
    self.descriptor["question_modifie"] = X
    
def modif_corpus(corpus,question_type,suppr_debut = True,suppr_interrogation = True):
    Lreturn = []
    for id_question in range(len(corpus)):
        text_app = corpus[id_question]
        if suppr_interrogation :
            lengh_question = len(text_app)
            text_app = text_app[:lengh_question-1]
        if suppr_debut and question_type[id_question]!='none of the above':
            text = text_app.split(' ')[2:]
            text_return = ''
            for i in text:
                text_return += " " +i
        Lreturn.append(text_app)
    return Lreturn


from torch.utils.data import DataLoader

# Précisez la localisation de vos données sur Google Drive
path = "harispe"
image_folder = "boolean_answers_dataset_images_10000"
descriptor = "boolean_answers_dataset_10000.csv"

batch_size = 16

# exemples de transformations
transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),     
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

vqa_dataset = VQADataset(path, descriptor, image_folder, transform=transform,vectorizer = TfidfVectorizer())
train , test = torch.utils.data.random_split(vqa_dataset,[9500,500])

train_set = DataLoader(train,batch_size=batch_size, shuffle=True, num_workers=0)
test_set = DataLoader(test,batch_size=batch_size, shuffle=True, num_workers=0)


count=0
for batch_id, batch in enumerate(train_set):
  if count%1000==0:
      print(count)
  count+=1
  break




#%% Définition du modèle 
from torchvision import models
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch.nn.functional as F

#from transformers import BertForSequenceClassification
#bert=BertForSequenceClassification.from_pretrained('bert-base-uncased')



""" D'un côté on utilise un CNN (par exemple Resnet18) pour traiter les images
    De l'autre on utilise un modèle type "DistilBert" pour traiter le texte
    On concatène les deux sorties pour les mettre en entrée d'un MLP
"""  
class VQA_model(torch.nn.Module):
    def __init__(self):
        super(VQA_model,self).__init__()
        self.resnet18=models.resnet18(pretrained=True)
        self.resnet18.fc=Identity() # 512 features en sortie
        
        self.distilbert=DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        #8self.token = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.distilbert.classifier=Identity() # 768 features en sortie
        
        self.input_lin = 512 + 768 # On concatène, la dimension est la somme
        self.lin1 = torch.nn.Linear(self.input_lin,1000)
        self.lin2 = torch.nn.Linear(1000,1000)
        self.lin3 = torch.nn.Linear(1000,1000)
        self.lin4 = torch.nn.Linear(1000,1000)
        self.lin5 = torch.nn.Linear(1000,2)
        
    def forward(self,image,question):
        representation_image = self.resnet18(image) #vecteur de taille 512
        
        
        #input_distil = self.token(question,return_tensors="pt",truncation=True, padding=True)
        output_distil = self.distilbert(**question)
        representation_texte = output_distil.logits #vecteur de taille 768
        
        X = torch.cat((representation_image,representation_texte),dim=1) #vecteur de taille 512+768=1280
        X = F.relu(self.lin1(X))
        X = F.relu(self.lin2(X))
        X = F.relu(self.lin3(X))
        X = F.relu(self.lin4(X))
        
        X = self.lin5(X)
        
        return X

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

        
model=VQA_model()  

print("nb of parameters",sum(p.data.nelement() for p in model.parameters()))
model.to("cpu")
token = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
question=batch[1]
question = token(question,return_tensors="pt",truncation=True, padding=True)
image=batch[0]

out=model(image,question)

sf_y_pred = torch.nn.Softmax(dim=1)(out) # softmax to obtain the probability distribution
_, predicted = torch.max(sf_y_pred , 1)

#%% Train et Test     
def train_optim(model, epochs, log_frequency, device, learning_rate):
  token = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
  model.to(device) # we make sure the model is on the proper device

  # Multiclass classification setting, we use cross-entropy
  # note that this implementation requires the logits as input 
  # logits: values prior softmax transformation 
  loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
  for t in range(epochs):

      model.train() # we specify that we are training the model

      # At each epoch, the training set will be processed as a set of batches
      for batch_id,  batch in enumerate(train_set) : 

        images, question, labels  = batch
        
        question = token(question,return_tensors="pt",truncation=True, padding=True)

        # we put the data on the same device
        images , question , labels = images.to(device), question.to(device) , labels.to(device)  
        
        y_pred = model(images,question) # forward pass output=logits

        loss = loss_fn(y_pred, labels)

        if batch_id % log_frequency == 0:
            print("epoch: {:03d}, batch: {:03d}, loss: {:.3f} ".format(t+1, batch_id+1, loss.item()))

        optimizer.zero_grad() # clear the gradient before backward
        loss.backward()       # update the gradient

        optimizer.step() # update the model parameters using the gradient

      # Model evaluation after each step computing the accuracy
      model.eval()
      total = 0
      correct = 0
      for batch_id, batch in enumerate(test_set):
        images , question , labels = batch
        question = token(question,return_tensors="pt",truncation=True, padding=True)
        images , question , labels = images.to(device), question.to(device) , labels.to(device) 
        y_pred = model(images,question) # forward computes the logits
        sf_y_pred = torch.nn.Softmax(dim=1)(y_pred) # softmax to obtain the probability distribution
        _, predicted = torch.max(sf_y_pred , 1)     # decision rule, we select the max
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
      print("[validation] accuracy: {:.3f}%\n".format(100 * correct / total))     
  return
epochs=100
log_frequency=300
device=torch.device("cuda:0")
learning_rate=1e-6   
      
train_optim(model,epochs,log_frequency,device,learning_rate)        
        
       
        
#%% Modèle 2

""" On construit seulement le MLP et on n'entraîne pas ni le resnet ni le distilbert

"""
class VQA_model_2(torch.nn.Module):
    def __init__(self):
        super(VQA_model_2,self).__init__()

        self.input_lin = 512 + 768 # On concatène, la dimension est la somme
        self.lin1 = torch.nn.Linear(self.input_lin,1000)
        self.lin2 = torch.nn.Linear(1000,1000)
        self.lin3 = torch.nn.Linear(1000,1000)
        self.lin4 = torch.nn.Linear(1000,1000)
        self.lin5 = torch.nn.Linear(1000,1000)
        self.lin_out = torch.nn.Linear(1000,2)
        
    def forward(self,X):
        X = F.relu(self.lin1(X))
        X = F.relu(self.lin2(X))
        X = F.relu(self.lin3(X))
        X = F.relu(self.lin4(X))
        X = F.relu(self.lin5(X))
        X = self.lin_out(X)
        
        return X    
        
model_2=VQA_model_2()  

print("nb of parameters",sum(p.data.nelement() for p in model_2.parameters()))        
        
def train_optim_2(model, epochs, log_frequency, device, learning_rate):
  
  resnet18=models.resnet18(pretrained=True)
  resnet18.fc=Identity()
  resnet18.to(device)
  
  distilbert=DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
  token = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
  distilbert.classifier=Identity()
  distilbert.to(device)
      
  model.to(device) # we make sure the model is on the proper device

  # Multiclass classification setting, we use cross-entropy
  # note that this implementation requires the logits as input 
  # logits: values prior softmax transformation 
  loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
  for t in range(epochs):

      model.train() # we specify that we are training the model

      # At each epoch, the training set will be processed as a set of batches
      for batch_id,  batch in enumerate(train_set) : 

        images, question, labels  = batch
        
        question = token(question,return_tensors="pt",truncation=True, padding=True)

        # we put the data on the same device
        images , question , labels = images.to(device), question.to(device) , labels.to(device)  
        
        representation_image = resnet18(images) #vecteur de taille 512
        output_distil = distilbert(**question)
        representation_texte = output_distil.logits #vecteur de taille 768
        
        X = torch.cat((representation_image,representation_texte),dim=1)
        
        y_pred = model(X) # forward pass output=logits

        loss = loss_fn(y_pred, labels)

        if batch_id % log_frequency == 0:
            print("epoch: {:03d}, batch: {:03d}, loss: {:.3f} ".format(t+1, batch_id+1, loss.item()))

        optimizer.zero_grad() # clear the gradient before backward
        loss.backward()       # update the gradient

        optimizer.step() # update the model parameters using the gradient

      # Model evaluation after each step computing the accuracy
      model.eval()
      total = 0
      correct = 0
      for batch_id, batch in enumerate(test_set):
        images , question , labels = batch
        question = token(question,return_tensors="pt",truncation=True, padding=True)
        images , question , labels = images.to(device), question.to(device) , labels.to(device) 
        
        representation_image = resnet18(images) #vecteur de taille 512
        output_distil = distilbert(**question)
        representation_texte = output_distil.logits #vecteur de taille 768
        
        X = torch.cat((representation_image,representation_texte),dim=1)
        y_pred = model(X) # forward computes the logits
        sf_y_pred = torch.nn.Softmax(dim=1)(y_pred) # softmax to obtain the probability distribution
        _, predicted = torch.max(sf_y_pred , 1)     # decision rule, we select the max
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
      print("[validation] accuracy: {:.3f}%\n".format(100 * correct / total))     
  return
epochs=2
log_frequency=100
device=torch.device("cuda:0")
learning_rate=1e-6   
      
#train_optim_2(model_2,epochs,log_frequency,device,learning_rate)        
        
        
        
        
        
        
        
        
        





