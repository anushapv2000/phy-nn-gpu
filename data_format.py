
# ----------------------------------------------------------------
# A class to format data
# Author:Anusha
# Email:Anusha
# ----------------------------------------------------------------
import os
import json
import uproot
import awkward
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

workpath=os.getcwd()

class DataFormat:

  def __init__(self,  split=True, config=None):
   #""" description """
   self.split=split
   self.config = config
  
   
  def load_data(self):
   if self.split:
     self.split_diff_file()
     
  def split_diff_file(self):
   tree_train=uproot.open(workpath+'/'+'data/train_D02kpipi0vxVc-cont0p5.root')['d0tree']
   tree_test=uproot.open(workpath+'/'+'data/test_D02kpipi0vxVc-cont0p5.root')['d0tree']
   df_train=tree_train.arrays(library="pd")
   df_test=tree_test.arrays(library="pd")

   df_train=df_train.drop(self.config["drop_feat"],axis=1)
   df_test=df_test.drop(self.config["drop_feat"], axis=1)
   print('df_test.tail(), type(df_test)')
   X_train=df_train.drop(self.config["targ_feat"],axis=1)
   y_train=df_train[self.config["targ_feat"]]
   X_test  =df_test.drop(self.config["targ_feat"],axis=1)
   y_test=df_test[self.config["targ_feat"]]
  
   """
   X_train=X_train.values
   X_test=X_test.values
   y_test=y_test.values
   y_train=y_train.values
   """

   sc=StandardScaler()
   X_train=sc.fit_transform(X_train.values)
   X_test=sc.transform(X_test.values)
   
   X_train=torch.FloatTensor(X_train).cuda()
   X_test=torch.FloatTensor(X_test).cuda()
   Y_train=torch.CharTensor(y_train.values).cuda()
   Y_test=torch.CharTensor(y_test.values).cuda()
   # print(type(X_train), type(X_train[0][0].item()))
   print('memory',torch.cuda.memory_allocated() / 1024**2)
   
   
with open("class_config.json", "r") as json_file:
 config=json.load(json_file)
 data_format = []
 for key,config in config.items():
   print(key)
   obj=DataFormat(True, config) 
   obj.load_data()
   data_format.append(obj)

print(data_format, type(data_format), type(data_format[0]))
