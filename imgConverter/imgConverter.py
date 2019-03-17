from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
import os
import numpy as np
import re

def imgConverter(dir = "[PATH TO MAP WITH IMAGES]"):
  regexp = re.compile("(atelectasis|infiltration|nodule)")

  first = True
  array = []
  names = []
  diseases = []
  diseaseDict = getDDict()
  for filename in os.listdir(dir):
      if filename.endswith(".png"):
        img = load_img(dir+os.sep+filename,target_size=(1024,1024))
        img = img_to_array(img)

        names.append(filename)
        disease = diseaseDict[filename]

        if first:
          first = False
          array = img
        else:
          array  = np.concatenate([[array],[img]])

        if len(disease) > 1:
          for i in range(2,len(disease)+1):
            if regexp.search(disease[i]):
              array = np.concatenate([[array],[img]])
              names.append(filename)
              diseases = diseases + [disease[i]]
        else:
          diseases = diseases + disease

  return array, names, diseases

def getDDict():
  file = open(r"imgNamesDiseases.csv")
  dict = {}
  for rule in file:
    part = rule.split(",")
    dict[part[0]] = part[1].split("|")

  return dict

