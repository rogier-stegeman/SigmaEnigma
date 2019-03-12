from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
import os

def imgConverter(dir = "informaticaProject"):
  first = True
  array = []
  for filename in os.listdir(dir):
      if filename.endswith(".png"):
        img = load_img(filename,target_size=(1024,1024))
        img = img_to_array(img)
        img = img.reshape((1,img.shape[1],img.shape[2],img.shape[0]))
        if first:
          first = False
          array = [img]
        else:
          array  = [array,img]
  return array

